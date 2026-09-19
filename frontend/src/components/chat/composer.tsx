import React from "react";
import { ComposerPrimitive } from "@assistant-ui/react";
import { ArrowUp } from "lucide-react";

export const Composer: React.FC = () => {
  return (
    <div className="sticky bottom-0 bg-gradient-to-t from-[#f8fafc] via-[#f8fafc]/95 to-transparent pt-3 pb-4 sm:pb-5 px-4 sm:px-6 z-10">
      <div className="max-w-3xl mx-auto">
        <ComposerPrimitive.Root className="relative flex items-center rounded-2xl border border-slate-300/90 bg-white shadow-sm focus-within:border-slate-600 focus-within:ring-3 focus-within:ring-slate-900/5 transition-all">
          <ComposerPrimitive.Input
            autoFocus
            rows={1}
            maxRows={6}
            placeholder="Ask about courses, billing, access, certificates…"
            className="w-full resize-none bg-transparent py-3.5 pl-4 pr-12 text-sm text-slate-900 placeholder:text-slate-400 focus:outline-hidden leading-relaxed"
          />

          <div className="absolute right-2.5 bottom-2.5">
            <ComposerPrimitive.Send
              className="w-8 h-8 rounded-xl bg-slate-900 text-white flex items-center justify-center hover:bg-slate-800 disabled:opacity-25 disabled:cursor-not-allowed transition-all shadow-xs cursor-pointer"
              aria-label="Send message"
            >
              <ArrowUp className="w-4 h-4" />
            </ComposerPrimitive.Send>
          </div>
        </ComposerPrimitive.Root>

        <p className="text-[11px] text-slate-500 text-center mt-2 select-none">
          Answers are generated from LearnForge support documentation. Account-specific requests may require support review.
        </p>
      </div>
    </div>
  );
};

