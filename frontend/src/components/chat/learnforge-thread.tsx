import React from "react";
import { ThreadPrimitive } from "@assistant-ui/react";
import { WelcomeState } from "./welcome-state";
import { UserMessage } from "./user-message";
import { AssistantMessage } from "./assistant-message";
import { Composer } from "./composer";

export const LearnForgeThread: React.FC = () => {
  return (
    <ThreadPrimitive.Root className="flex-1 flex flex-col min-h-0 bg-[#f8fafc]">
      <ThreadPrimitive.Viewport className="flex-1 overflow-y-auto min-h-0 flex flex-col">
        <div className="w-full max-w-3xl mx-auto flex-1 flex flex-col py-4">
          <ThreadPrimitive.Empty>
            <WelcomeState />
          </ThreadPrimitive.Empty>

          <ThreadPrimitive.Messages
            components={{
              UserMessage,
              AssistantMessage,
            }}
          />
        </div>
      </ThreadPrimitive.Viewport>

      <Composer />
    </ThreadPrimitive.Root>
  );
};

