import React from "react";
import { FileText } from "lucide-react";
import type { SourceMessagePartProps } from "@assistant-ui/core/react";

export const SourceChip: React.FC<SourceMessagePartProps> = (props) => {
  const id = props.id;
  const title = props.title || id;

  return (
    <div
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
};

