"use client";

import CodeViewer from "@/components/code-viewer";
import { useScrollTo } from "@/hooks/use-scroll-to";
import { CheckIcon } from "@heroicons/react/16/solid";
import { ArrowLongRightIcon, ChevronDownIcon } from "@heroicons/react/20/solid";
import { ArrowUpOnSquareIcon } from "@heroicons/react/24/outline";
import * as Select from "@radix-ui/react-select";
import * as Switch from "@radix-ui/react-switch";
import { AnimatePresence, motion } from "framer-motion";
import { FormEvent, useEffect, useState } from "react";
import LoadingDots from "../../components/loading-dots";

function removeCodeFormatting(code: string): string {
  return code.replace(/```(?:typescript|javascript|tsx)?\n([\s\S]*?)```/g, '$1').trim();
}

export default function Home() {
  let [status, setStatus] = useState<
    "initial" | "creating" | "created" | "updating" | "updated"
  >("initial");
  let [prompt, setPrompt] = useState("");
  let models = [
    {
      label: "gemini-2.0-flash-exp",
      value: "gemini-2.0-flash-exp",
    },
    {
      label: "gemini-1.5-pro",
      value: "gemini-1.5-pro",
    },
    {
      label: "gemini-1.5-flash",
      value: "gemini-1.5-flash",
    },
    {
      label: "gemini-2.5-pro-exp",
      value: "gemini-2.5-pro-exp",
    },
    {
      label: "openai/gpt-4o",
      value: "openai/gpt-4o",
    },
    {
      label: "openai/gpt-4-turbo",
      value: "openai/gpt-4-turbo",
    },
    {
      label: "anthropic/claude-3.5-sonnet",
      value: "anthropic/claude-3.5-sonnet",
    },
    {
      label: "anthropic/claude-3-opus",
      value: "anthropic/claude-3-opus",
    },
  ];
  let [model, setModel] = useState(models[0].value);
  let [modification, setModification] = useState("");
  let [generatedCode, setGeneratedCode] = useState("");
  let [initialAppConfig, setInitialAppConfig] = useState({
    model: "",
  });
  let [ref, scrollTo] = useScrollTo();
  let [messages, setMessages] = useState<{ role: string; content: string }[]>(
    [],
  );

  let loading = status === "creating" || status === "updating";

  async function createApp(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();

    if (status !== "initial") {
      scrollTo({ delay: 0.5 });
    }

    setStatus("creating");
    setGeneratedCode("");

    let res = await fetch("/api/generateCode", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model,
        messages: [{ role: "user", content: prompt }],
      }),
    });

    if (!res.ok) {
      throw new Error(res.statusText);
    }

    if (!res.body) {
      throw new Error("No response body");
    }

    const reader = res.body.getReader();
    let receivedData = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      receivedData += new TextDecoder().decode(value);
      const cleanedData = removeCodeFormatting(receivedData);
      setGeneratedCode(cleanedData);
    }

    setMessages([{ role: "user", content: prompt }]);
    setInitialAppConfig({ model });
    setStatus("created");
  }

  useEffect(() => {
    let el = document.querySelector(".cm-scroller");
    if (el && loading) {
      let end = el.scrollHeight - el.clientHeight;
      el.scrollTo({ top: end });
    }
  }, [loading, generatedCode]);

  return (
    <main className="mt-12 flex w-full flex-1 flex-col items-center px-4 text-center sm:mt-1">
      <a
        className="mb-4 inline-flex h-7 shrink-0 items-center gap-[9px] rounded-[50px] border-[0.5px] border-solid border-[#E6E6E6] bg-[rgba(234,238,255,0.65)] dark:bg-[rgba(30,41,59,0.5)] dark:border-gray-700 px-7 py-5 shadow-[0px_1px_1px_0px_rgba(0,0,0,0.25)]"
        href="https://ai.google.dev/gemini-api/docs"
        target="_blank"
      >
        <span className="text-center">
          Powered by <span className="font-medium">Gemini API</span>
        </span>
      </a>
      <h1 className="my-6 max-w-3xl text-4xl font-bold text-gray-800 dark:text-white sm:text-6xl">
        Turn your <span className="text-blue-600">idea</span>
        <br /> into an <span className="text-blue-600">app</span>
      </h1>

      <form className="w-full max-w-xl" onSubmit={createApp}>
        <fieldset disabled={loading} className="disabled:opacity-75">
          <div className="relative mt-5">
            <div className="absolute -inset-2 rounded-[32px] bg-gray-300/50 dark:bg-gray-800/50" />
            <div className="relative flex rounded-3xl bg-white dark:bg-[#1E293B] shadow-sm">
              <div className="relative flex flex-grow items-stretch focus-within:z-10">
                <textarea
                  rows={3}
                  required
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  name="prompt"
                  className="w-full resize-none rounded-l-3xl bg-transparent px-6 py-5 text-lg focus-visible:outline focus-visible:outline-2 focus-visible:outline-blue-500 dark:text-gray-100 dark:placeholder-gray-400"
                  placeholder="Build me a calculator app..."
                />
              </div>
              <button
                type="submit"
                disabled={loading}
                className="relative -ml-px inline-flex items-center gap-x-1.5 rounded-r-3xl px-3 py-2 text-sm font-semibold text-blue-500 hover:text-blue-400 focus-visible:outline focus-visible:outline-2 focus-visible:outline-blue-500 disabled:text-gray-900 dark:disabled:text-gray-400"
              >
                {status === "creating" ? (
                  <LoadingDots color="black" style="large" />
                ) : (
                  <ArrowLongRightIcon className="-ml-0.5 size-6" />
                )}
              </button>
            </div>
          </div>
          <div className="mt-6 flex flex-col justify-center gap-4 sm:flex-row sm:items-center sm:gap-8">
            <div className="flex items-center justify-between gap-3 sm:justify-center">
              <p className="text-gray-500 dark:text-gray-400 sm:text-xs">Model:</p>
              <Select.Root
                name="model"
                disabled={loading}
                value={model}
                onValueChange={(value) => setModel(value)}
              >
                <Select.Trigger className="group flex w-60 max-w-xs items-center rounded-2xl border-[6px] border-gray-300 dark:border-gray-700 bg-white dark:bg-[#1E293B] px-4 py-2 text-sm focus-visible:outline focus-visible:outline-2 focus-visible:outline-blue-500">
                  <Select.Value />
                  <Select.Icon className="ml-auto">
                    <ChevronDownIcon className="size-6 text-gray-300 group-focus-visible:text-gray-500 group-enabled:group-hover:text-gray-500 dark:text-gray-600 dark:group-focus-visible:text-gray-400 dark:group-enabled:group-hover:text-gray-400" />
                  </Select.Icon>
                </Select.Trigger>
                <Select.Portal>
                  <Select.Content className="overflow-hidden rounded-md bg-white dark:bg-[#1E293B] shadow-lg">
                    <Select.Viewport className="p-2">
                      {models.map((model) => (
                        <Select.Item
                          key={model.value}
                          value={model.value}
                          className="flex cursor-pointer items-center rounded-md px-3 py-2 text-sm data-[highlighted]:bg-gray-100 dark:data-[highlighted]:bg-gray-800 data-[highlighted]:outline-none"
                        >
                          <Select.ItemText asChild>
                            <span className="inline-flex items-center gap-2 text-gray-500 dark:text-gray-400">
                              <div className="size-2 rounded-full bg-green-500" />
                              {model.label}
                            </span>
                          </Select.ItemText>
                          <Select.ItemIndicator className="ml-auto">
                            <CheckIcon className="size-5 text-blue-600" />
                          </Select.ItemIndicator>
                        </Select.Item>
                      ))}
                    </Select.Viewport>
                    <Select.ScrollDownButton />
                    <Select.Arrow />
                  </Select.Content>
                </Select.Portal>
              </Select.Root>
            </div>
          </div>
        </fieldset>
      </form>

      <hr className="border-1 mb-20 h-px bg-gray-700 dark:bg-gray-700/30" />

      {status !== "initial" && (
        <motion.div
          initial={{ height: 0 }}
          animate={{
            height: "auto",
            overflow: "hidden",
            transitionEnd: { overflow: "visible" },
          }}
          transition={{ type: "spring", bounce: 0, duration: 0.5 }}
          className="w-full pb-[25vh] pt-1"
          onAnimationComplete={() => scrollTo()}
          ref={ref}
        >
          <div className="relative mt-8 w-full overflow-hidden">
            <div className="isolate">
              <CodeViewer code={generatedCode} showEditor />
            </div>

            <AnimatePresence>
              {loading && (
                <motion.div
                  initial={status === "updating" ? { x: "100%" } : undefined}
                  animate={status === "updating" ? { x: "0%" } : undefined}
                  exit={{ x: "100%" }}
                  transition={{
                    type: "spring",
                    bounce: 0,
                    duration: 0.85,
                    delay: 0.5,
                  }}
                  className="absolute inset-x-0 bottom-0 top-1/2 flex items-center justify-center rounded-r border border-gray-400 dark:border-gray-700 bg-gradient-to-br from-gray-100 to-gray-300 dark:from-[#1E293B] dark:to-gray-800 md:inset-y-0 md:left-1/2 md:right-0"
                >
                  <p className="animate-pulse text-3xl font-bold dark:text-gray-100">
                    {status === "creating"
                      ? "Building your app..."
                      : "Updating your app..."}
                  </p>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </motion.div>
      )}
    </main>
  );
}

async function minDelay<T>(promise: Promise<T>, ms: number) {
  let delay = new Promise((resolve) => setTimeout(resolve, ms));
  let [p] = await Promise.all([promise, delay]);

  return p;
}
