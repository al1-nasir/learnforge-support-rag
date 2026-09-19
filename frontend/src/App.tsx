import React from "react";
import { AssistantRuntimeProvider } from "@assistant-ui/react";
import { useLearnForgeRuntime } from "./runtime/learnforge-runtime";
import { Header } from "./components/chat/header";
import { LearnForgeThread } from "./components/chat/learnforge-thread";

export const App: React.FC = () => {
  const runtime = useLearnForgeRuntime();

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <div className="flex flex-col h-screen bg-[#f8fafc] text-slate-900 selection:bg-slate-900 selection:text-white">
        <Header />
        <main className="flex-1 flex flex-col min-h-0">
          <LearnForgeThread />
        </main>
      </div>
    </AssistantRuntimeProvider>
  );
};

export default App;
