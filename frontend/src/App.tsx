import { Navigate, Route, Routes } from "react-router-dom";
import { AppLayout } from "./components/AppLayout";
import { KnowledgeBaseProvider } from "./context/KnowledgeBaseContext";
import { ModelProvider } from "./context/ModelContext";
import { ProgrammingAssistant } from "./sections/ProgrammingAssistant";
import { QuizPrep } from "./sections/QuizPrep";
import { RegularChat } from "./sections/RegularChat";
import { StudyAgent } from "./sections/StudyAgent";

export function App() {
  return (
    <ModelProvider>
      <KnowledgeBaseProvider>
        <Routes>
          <Route element={<AppLayout />}>
            <Route index element={<Navigate to="/quiz" replace />} />
            <Route path="/quiz" element={<QuizPrep />} />
            <Route path="/agent" element={<StudyAgent />} />
            <Route path="/code" element={<ProgrammingAssistant />} />
            <Route path="/chat" element={<RegularChat />} />
            <Route path="*" element={<Navigate to="/quiz" replace />} />
          </Route>
        </Routes>
      </KnowledgeBaseProvider>
    </ModelProvider>
  );
}
