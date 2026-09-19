import React from "react";
import { MessagePrimitive } from "@assistant-ui/react";

export const UserMessage: React.FC = () => {
  return (
    <MessagePrimitive.Root className="flex justify-end my-3 px-4 sm:px-6">
      <div className="max-w-[85%] sm:max-w-[75%] rounded-2xl rounded-tr-xs bg-slate-900 px-4 py-2.5 text-sm text-white shadow-xs leading-relaxed break-words font-normal">
        <MessagePrimitive.Content />
      </div>
    </MessagePrimitive.Root>
  );
};

