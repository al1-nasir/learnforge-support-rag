import React from "react";
import { ThreadPrimitive } from "@assistant-ui/react";
import { RefreshCcw, Download, Award, CreditCard } from "lucide-react";

const SUGGESTIONS = [
  {
    icon: RefreshCcw,
    text: "How do course refunds work?",
    subtext: "Eligibility window and terms",
  },
  {
    icon: Download,
    text: "Can I watch courses offline?",
    subtext: "Mobile and desktop offline options",
  },
  {
    icon: Award,
    text: "Where is my certificate?",
    subtext: "Completion requirements & access",
  },
  {
    icon: CreditCard,
    text: "I don't recognize a charge",
    subtext: "Pending holds and renewals",
  },
];

export const WelcomeState: React.FC = () => {
  return (
    <div className="flex flex-col items-center justify-center text-center px-4 py-10 sm:py-16 max-w-xl mx-auto my-auto">
      <div className="w-12 h-12 rounded-2xl bg-slate-900 text-white flex items-center justify-center font-bold text-lg mb-4 shadow-sm select-none">
        LF
      </div>
      <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-900 mb-1.5">
        LearnForge Support
      </h2>
      <p className="text-base sm:text-lg font-medium text-slate-700 mb-2">
        How can we help?
      </p>
      <p className="text-sm text-slate-500 max-w-md mb-8 leading-relaxed">
        Ask about course access, billing, refunds, certificates, account issues, or using LearnForge.
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 w-full text-left">
        {SUGGESTIONS.map((item) => (
          <ThreadPrimitive.Suggestion
            key={item.text}
            prompt={item.text}
            send={true}
            className="flex items-start gap-3 p-3.5 rounded-xl border border-slate-200/90 bg-white hover:bg-slate-50/80 hover:border-slate-300 transition-all text-left shadow-xs group cursor-pointer"
          >
            <item.icon className="w-4 h-4 text-slate-400 group-hover:text-slate-700 mt-0.5 flex-shrink-0 transition-colors" />
            <div className="flex-1 min-w-0">
              <div className="text-sm font-medium text-slate-800 leading-snug group-hover:text-slate-900">
                "{item.text}"
              </div>
              <div className="text-xs text-slate-400 mt-0.5 truncate">
                {item.subtext}
              </div>
            </div>
          </ThreadPrimitive.Suggestion>
        ))}
      </div>
    </div>
  );
};

