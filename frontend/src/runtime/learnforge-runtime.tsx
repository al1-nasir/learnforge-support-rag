import { useMemo, useRef } from "react";
import {
  useLocalRuntime,
  type ChatModelAdapter,
  type ThreadAssistantMessagePart,
} from "@assistant-ui/react";
import { sendChatMessage } from "../lib/api";
import type { SupportMeta } from "../lib/types";

export function useLearnForgeRuntime() {
  // Single session ID preserved in React memory for the lifetime of this page
  const sessionIdRef = useRef<string>(crypto.randomUUID());

  const adapter = useMemo<ChatModelAdapter>(() => {
    return {
      async run({ messages, abortSignal }) {
        // Extract the latest user message
        const lastUserMsg = [...messages].reverse().find((m) => m.role === "user");
        let userText = "";
        if (lastUserMsg) {
          for (const part of lastUserMsg.content) {
            if (part.type === "text") {
              userText += part.text;
            }
          }
        }

        const trimmed = userText.trim();
        if (!trimmed) {
          return { content: [] };
        }

        try {
          const response = await sendChatMessage(
            {
              session_id: sessionIdRef.current,
              message: trimmed,
            },
            abortSignal,
          );


          // Map decision, escalation metadata, and citations to custom data part
          const metaData: SupportMeta = {
            decision: response.decision,
            reasonCode: response.reason_code,
            handoffSummary: response.handoff_summary,
            citations: response.citations || [],
          };

          const metaPart: ThreadAssistantMessagePart = {
            type: "data" as const,
            name: "support-meta",
            data: metaData,
          };

          return {
            content: [
              {
                type: "text" as const,
                text: response.message,
              },
              metaPart,
            ],
          };
        } catch (err: unknown) {
          console.error("LearnForge Support chat request error:", err);
          return {
            content: [
              {
                type: "text" as const,
                text: "LearnForge Support is temporarily unavailable. Please try again.",
              },
            ],
            status: {
              type: "incomplete" as const,
              reason: "error" as const,
              error: err instanceof Error ? err.message : "Connection failure",
            },
          };
        }
      },
    };
  }, []);

  return useLocalRuntime(adapter);
}

