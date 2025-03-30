import dedent from "dedent";
import { z } from "zod";
import { GoogleGenerativeAI } from "@google/generative-ai";
import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";

// Initialize Google Gemini
const googleApiKey = process.env.GOOGLE_AI_API_KEY || "";
const genAI = new GoogleGenerativeAI(googleApiKey);

// Initialize OpenAI
const openaiApiKey = process.env.OPENAI_API_KEY || "";
const openai = new OpenAI({ apiKey: openaiApiKey });

// Initialize Anthropic
const anthropicApiKey = process.env.ANTHROPIC_API_KEY || "";
const anthropic = new Anthropic({ apiKey: anthropicApiKey });

const RequestBodySchema = z.object({
  model: z.string(),
  messages: z.array(
    z.object({
      role: z.enum(["user", "assistant"]), // Anthropic uses 'assistant', OpenAI uses 'assistant', Gemini uses 'model' but we can map
      content: z.string(),
    }),
  ),
});

export async function POST(req: Request) {
  let json: unknown;
  try {
    json = await req.json();
  } catch (error) {
    return new Response("Invalid JSON input", { status: 400 });
  }

  const result = RequestBodySchema.safeParse(json);

  if (result.error) {
    return new Response(result.error.message, { status: 422 });
  }

  let { model: modelId, messages } = result.data;
  let systemPrompt = getSystemPrompt();
  // Assuming the last message is the user's prompt
  const userPrompt = messages[messages.length - 1].content;

  // Combine system prompt and user prompt (adjust format as needed per provider)
  const fullPrompt = `${systemPrompt}\n\nUser Prompt:\n${userPrompt}\n\nPlease ONLY return code, NO backticks or language names. Don't start with \`\`\`typescript or \`\`\`javascript or \`\`\`tsx or \`\`\`.`;

  // --- Provider Logic ---
  try {
    let stream: AsyncIterable<any>;
    let responseStream: ReadableStream<Uint8Array>;

    if (modelId.startsWith("openai/")) {
      const openaiModelId = modelId.replace("openai/", "");
      stream = await openai.chat.completions.create({
        model: openaiModelId,
        messages: [ // OpenAI prefers system message separate
           { role: "system", content: systemPrompt },
           ...messages.map(m => ({ role: m.role as 'user' | 'assistant', content: m.content }))
        ],
        stream: true,
      });

      responseStream = new ReadableStream({
        async start(controller) {
          const encoder = new TextEncoder();
          for await (const chunk of stream) {
            const content = chunk.choices[0]?.delta?.content || "";
            if (content) {
              controller.enqueue(encoder.encode(content));
            }
          }
          controller.close();
        },
      });

    } else if (modelId.startsWith("anthropic/")) {
      const anthropicModelId = modelId.replace("anthropic/", "");
       // Anthropic expects system prompt at a specific top-level parameter
       // And messages alternate user/assistant
      const anthropicMessages = messages.filter(m => m.role === 'user'); // Simple example, might need more complex history management

      stream = await anthropic.messages.create({
        model: anthropicModelId,
        system: systemPrompt, // System prompt here
        messages: anthropicMessages.map(m => ({ role: m.role, content: m.content })),
        max_tokens: 4096, // Example max_tokens, adjust as needed
        stream: true,
      });

      responseStream = new ReadableStream({
          async start(controller) {
            const encoder = new TextEncoder();
            for await (const event of stream) {
              if (event.type === 'content_block_delta' && event.delta.type === 'text_delta') {
                controller.enqueue(encoder.encode(event.delta.text));
              } else if (event.type === 'message_delta' && event.delta.stop_reason) {
                 // Handle stop reason if needed, e.g., logging
              } else if (event.type === 'message_stop') {
                 // Message finished
              }
            }
            controller.close();
          }
        });

    } else { // Assume Google Gemini
      const geminiModel = genAI.getGenerativeModel({ model: modelId });
      // Gemini combines system+user prompt (as done originally)
      const geminiStream = await geminiModel.generateContentStream(fullPrompt);

      responseStream = new ReadableStream({
        async start(controller) {
          const encoder = new TextEncoder();
          for await (const chunk of geminiStream.stream) {
             try {
                const chunkText = chunk.text();
                controller.enqueue(encoder.encode(chunkText));
             } catch (error) {
                console.error("Error processing Gemini chunk:", error);
                // Handle error, maybe close stream with error?
             }
          }
          controller.close();
        },
      });
    }

    return new Response(responseStream, {
        headers: { 'Content-Type': 'text/plain; charset=utf-8' },
    });

  } catch (error) {
    console.error("Error calling AI API:", error);
    let errorMessage = "Failed to generate code";
    if (error instanceof Error) {
        errorMessage = error.message;
    } else if (typeof error === 'string') {
        errorMessage = error;
    }
     // Check for specific API error types if needed (e.g. OpenAI.APIError)
    if (error instanceof OpenAI.APIError) {
        errorMessage = `OpenAI Error (${error.status}): ${error.message}`;
    } else if (error instanceof Anthropic.APIError) {
        errorMessage = `Anthropic Error (${error.status}): ${error.message}`;
    }
    // Consider checking for Google AI specific errors too

    return new Response(JSON.stringify({ error: errorMessage }), { status: 500, headers: { 'Content-Type': 'application/json'} });
  }
}


function getSystemPrompt() {
  // This prompt might need adjustment per provider for optimal results.
  // Keeping it generic for now.
  let systemPrompt =
`You are an expert frontend React engineer who is also a great UI/UX designer. Follow the instructions carefully:

- Think carefully step by step.
- Create a React component for whatever the user asked you to create and make sure it can run by itself by using a default export.
- Make sure the React app is interactive and functional by creating state when needed and having no required props.
- If you use any imports from React like useState or useEffect, make sure to import them directly.
- Use TypeScript as the language for the React component.
- Use Tailwind classes for styling. DO NOT USE ARBITRARY VALUES (e.g. \`h-[600px]\`). Use a consistent color palette and spacing.
- Please ONLY return the full React code starting with the imports, nothing else. It's very important for my job that you only return the React code with imports. DO NOT START WITH \`\`\`typescript or \`\`\`javascript or \`\`\`tsx or \`\`\`.
- ONLY IF the user asks for a dashboard, graph or chart, the recharts library is available to be imported, e.g. \`import { LineChart, XAxis, ... } from "recharts"\` & \`<LineChart ...><XAxis dataKey="name"> ...\`. Please only use this when needed.
- For placeholder images, please use a <div className="bg-gray-200 border-2 border-dashed rounded-xl w-16 h-16" />
- NO OTHER LIBRARIES (e.g. zod, hookform) ARE INSTALLED OR ABLE TO BE IMPORTED.
  `;
  return dedent(systemPrompt);
}

// Keep edge runtime for now, but monitor for compatibility issues with SDKs
export const runtime = "edge";