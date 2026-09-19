import React from "react";
import { MessagePrimitive, useAuiState } from "@assistant-ui/react";
import { MarkdownTextPrimitive } from "@assistant-ui/react-markdown";
import type { SupportMeta } from "../../lib/types";
import { SupportStatus } from "./support-status";

const MarkdownText: React.FC = () => {
  return (
    <MarkdownTextPrimitive
      smooth={false}
      className="text-sm text-slate-800 leading-relaxed break-words space-y-2.5 font-normal"
    />
  );
};

export const AssistantMessage: React.FC = () => {
  const isRunning = useAuiState((s) => s.message.status?.type === "running");
  const hasText = useAuiState((s) => {
    if (s.message.role !== "assistant") return false;
    return s.message.content.some(
      (p) => p.type === "text" && p.text.trim().length > 0
    );
  });

  return (
    <MessagePrimitive.Root className="flex items-start gap-3 my-4 px-4 sm:px-6">
      <div className="w-8 h-8 rounded-lg bg-slate-900 text-white flex items-center justify-center font-bold text-xs select-none flex-shrink-0 mt-0.5 shadow-xs">
        LF
      </div>

      <div className="flex-1 min-w-0 flex flex-col gap-2">
        {/* Loading indicator while waiting for response */}
        {isRunning && !hasText && (
          <div className="flex items-center gap-2 text-xs font-medium text-slate-500 py-1 select-none">
            <span className="flex gap-1 items-center">
              <span className="w-1.5 h-1.5 rounded-full bg-slate-400 animate-bounce [animation-delay:-0.3s]" />
              <span className="w-1.5 h-1.5 rounded-full bg-slate-400 animate-bounce [animation-delay:-0.15s]" />
              <span className="w-1.5 h-1.5 rounded-full bg-slate-400 animate-bounce" />
            </span>
            <span>Checking LearnForge support knowledge…</span>
          </div>
        )}

        {/* Text and custom metadata content */}
        <div className="text-slate-800">
          <MessagePrimitive.Content
            components={{
              Text: MarkdownText,
              Source: () => null,
              data: {
                by_name: {
                  "support-meta": ({ data }: { data: SupportMeta }) => (
                    <SupportStatus meta={data} />
                  ),
                },
                Fallback: () => null,
              },
            }}
          />
        </div>
      </div>
    </MessagePrimitive.Root>
  );
};


