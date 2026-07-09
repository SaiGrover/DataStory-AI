import { Navigate, Route, Routes } from "react-router-dom";
import { AppStateProvider } from "./lib/store";
import { Layout } from "./components/Layout";
import { StartPage } from "./pages/StartPage";
import { HealthPage } from "./pages/HealthPage";
import { CleanPage } from "./pages/CleanPage";
import { ExplorePage } from "./pages/ExplorePage";
import { ModelsPage } from "./pages/ModelsPage";
import { ResultsPage } from "./pages/ResultsPage";
import { ReportPage } from "./pages/ReportPage";

export default function App() {
  return (
    <AppStateProvider>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<StartPage />} />
          <Route path="/health" element={<HealthPage />} />
          <Route path="/clean" element={<CleanPage />} />
          <Route path="/explore" element={<ExplorePage />} />
          <Route path="/models" element={<ModelsPage />} />
          <Route path="/results" element={<ResultsPage />} />
          <Route path="/report" element={<ReportPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </AppStateProvider>
  );
}
