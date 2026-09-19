import React from "react";
import { AlertCircle, FileText } from "lucide-react";

interface EscalationCardProps {
  handoffSummary?: string | null;
  reasonCode?: string;
}

export const EscalationCard: React.FC<EscalationCardProps> = ({
  handoffSummary,
}) => {
  return (
    <div className="my-2 rounded-xl border border-amber-200/80 bg-amber-50/50 p-4 text-amber-950 shadow-xs">
      <div className="flex items-center gap-2 font-semibold text-amber-900 text-sm">
        <AlertCircle className="w-4 h-4 text-amber-600 flex-shrink-0" />
        <span>Support review required</span>
      </div>

      <p className="mt-1.5 text-xs text-amber-800 leading-relaxed">
        This request requires account-specific verification or intervention by LearnForge Support personnel.
      </p>

      {handoffSummary && (
        <div className="mt-3 pt-3 border-t border-amber-200/60">
          <div className="flex items-center gap-1.5 text-xs font-semibold text-amber-900 uppercase tracking-wider mb-1">
            <FileText className="w-3.5 h-3.5 text-amber-700" />
            <span>Handoff summary</span>
          </div>
          <div className="text-xs text-amber-900/90 font-mono bg-amber-100/40 p-2.5 rounded-lg border border-amber-200/50 leading-relaxed whitespace-pre-wrap">
            {handoffSummary}
          </div>
        </div>
      )}
    </div>
  );
};

