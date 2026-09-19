import React from "react";
import { CheckCircle2, Info, FileText } from "lucide-react";
import type { SupportMeta } from "../../lib/types";
import { EscalationCard } from "./escalation-card";

interface SupportStatusProps {
  meta: SupportMeta;
}

export const SupportStatus: React.FC<SupportStatusProps> = ({ meta }) => {
  const hasCitations = meta.citations && meta.citations.length > 0;

  return (
    <div className="flex flex-col gap-2.5 pt-1">
      {meta.decision === "escalate" ? (
        <EscalationCard
          handoffSummary={meta.handoffSummary}
          reasonCode={meta.reasonCode}
        />
      ) : meta.decision === "clarify" ? (
        <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium bg-sky-50 text-sky-700 border border-sky-200/70 w-fit select-none">
          <Info className="w-3.5 h-3.5 text-sky-600 flex-shrink-0" />
          <span>More information needed</span>
        </div>
      ) : (
        <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium bg-emerald-50 text-emerald-700 border border-emerald-200/70 w-fit select-none">
          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 flex-shrink-0" />
          <span>Grounded answer</span>
        </div>
      )}

      {hasCitations && (
        <div className="mt-1 pt-2 border-t border-slate-100 flex flex-col gap-1.5">
          <div className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider flex items-center gap-1.5">
            <span>Sources</span>
            <span className="px-1.5 py-0.5 rounded-full bg-slate-100 text-slate-600 text-[10px] font-medium">
              {meta.citations!.length}
            </span>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {meta.citations!.map((c, idx) => {
              const id = c.record_id;
              const title = c.title || id;
              return (
                <div
                  key={idx}
                  title={title !== id ? title : undefined}
                  className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium bg-slate-100/90 text-slate-700 border border-slate-200/80 hover:bg-slate-200/60 hover:text-slate-900 transition-colors select-none"
                >
                  <FileText className="w-3.5 h-3.5 text-slate-400 flex-shrink-0" />
                  <span className="font-semibold text-slate-800">{id}</span>
                  {title && title !== id && (
                    <span className="text-slate-500 max-w-[220px] truncate border-l border-slate-300 pl-1.5 hidden sm:inline">
                      {title}
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};

