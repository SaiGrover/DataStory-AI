import { ArrowRight, BarChart3, Bot, CloudUpload, Database, FlaskConical, LineChart, MessageSquare, Moon, Sun, Trophy, Wand2 } from "lucide-react";
import { Link, NavLink, Outlet } from "react-router-dom";
import { useAppState } from "../lib/store";

const navItems = [
  { to: "/", label: "Upload", description: "Start analysis", icon: CloudUpload },
  { to: "/health", label: "Data Health", description: "Check dataset", icon: Database },
  { to: "/clean", label: "Clean Data", description: "Fix & prepare", icon: Wand2 },
  { to: "/explore", label: "EDA", description: "Discover insights", icon: LineChart },
  { to: "/models", label: "Modeling", description: "Train & tune", icon: FlaskConical },
  { to: "/results", label: "Results", description: "Compare models", icon: Trophy },
  { to: "/agent", label: "Agent", description: "Ask dataset", icon: Bot },
  { to: "/report", label: "AI Report", description: "Generate report", icon: MessageSquare },
];

export function Layout() {
  const { dataset, darkMode, setDarkMode } = useAppState();
  return (
    <div className={darkMode ? "dark min-h-screen bg-[#0f1115] text-zinc-100" : "min-h-screen bg-milk text-ink"}>
      <header className="sticky top-0 z-40 border-b border-line bg-white/95 backdrop-blur dark:border-[#303844] dark:bg-[#0f1115]/92">
        <div className="mx-auto grid max-w-[1500px] grid-cols-[1fr_auto_1fr] items-center px-8 py-4">
          <Link to="/" className="flex items-center gap-3">
            <BarChart3 className="h-9 w-9 text-sage" />
            <div>
              <div className="text-3xl font-extrabold leading-none tracking-normal">DataStory AI</div>
            </div>
          </Link>
          <div className="text-center text-base font-semibold text-zinc-600 dark:text-[#b9c2ce]">Your Junior Data Analyst</div>
          <button className="justify-self-end ds-button-secondary" onClick={() => setDarkMode(!darkMode)} aria-label="Toggle dark mode">
            {darkMode ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            {darkMode ? "Light" : "Dark"}
          </button>
        </div>
        {dataset && (
          <div className="border-t border-line bg-milk px-8 py-2 text-sm text-zinc-600 dark:border-[#303844] dark:bg-[#151a21] dark:text-[#b9c2ce]">
            <div className="mx-auto flex max-w-[1500px] gap-4">
              <strong>{dataset.filename}</strong>
              <span>{dataset.rows.toLocaleString()} rows</span>
              <span>{dataset.columns.toLocaleString()} columns</span>
              <span>Health {dataset.profile.health_score ?? 0}/100</span>
            </div>
          </div>
        )}
      </header>
      <main className="mx-auto max-w-[1500px] px-8 pb-36 pt-7">
        <Outlet />
      </main>
      <nav className="fixed bottom-0 left-0 right-0 z-50 border-t border-line bg-white/95 px-3 py-3 shadow-[0_-10px_30px_rgba(39,43,34,.08)] backdrop-blur dark:border-[#303844] dark:bg-[#0f1115]/94 dark:shadow-[0_-14px_34px_rgba(0,0,0,.34)]">
        <div className="mx-auto flex max-w-[1460px] items-center justify-between gap-1 overflow-hidden">
          {navItems.map((item, index) => (
            <div key={item.to} className="flex min-w-0 flex-1 items-center">
              <NavLink
                to={item.to}
                className={({ isActive }) =>
                  `group grid min-w-0 flex-1 grid-cols-[40px_minmax(0,1fr)] items-center gap-2 rounded-xl px-1.5 py-1.5 transition ${
                    isActive ? "text-sage" : "text-zinc-500 hover:bg-stone-50 hover:text-ink dark:text-[#a7b0bc] dark:hover:bg-[#1d232c] dark:hover:text-zinc-100"
                  }`
                }
              >
                {({ isActive }) => (
                  <>
                    <span
                      className={`grid h-10 w-10 place-items-center rounded-full border transition ${
                        isActive
                          ? "border-sage bg-sage text-white dark:text-[#11151a]"
                          : "border-line bg-[#F0F1EC] text-zinc-500 group-hover:text-sage dark:border-[#394735] dark:bg-[#182118]"
                      }`}
                    >
                      <item.icon className="h-5 w-5" />
                    </span>
                    <span className="min-w-0">
                      <span className={`block whitespace-nowrap text-[13px] font-bold leading-4 ${isActive ? "text-sage" : "text-ink dark:text-zinc-100"}`}>{item.label}</span>
                      <span className="block truncate text-[11px] text-zinc-500 dark:text-[#9fa9b6]">
                        {index === 0 && dataset ? dataset.filename : item.description}
                      </span>
                    </span>
                  </>
                )}
              </NavLink>
              {index < navItems.length - 1 && (
                <div className="relative h-12 w-5 shrink-0" aria-hidden="true">
                  <div className="absolute left-0 right-1 top-1/2 h-px -translate-y-1/2 bg-gradient-to-r from-sage/20 via-sage/80 to-sage shadow-[0_0_14px_rgba(111,133,84,.75)]" />
                  <div className="absolute right-0 top-1/2 h-6 w-6 -translate-y-1/2 rounded-full bg-sage/20 blur-md" />
                  <ArrowRight className="absolute right-0 top-1/2 h-4 w-4 -translate-y-1/2 text-sage drop-shadow-[0_0_7px_rgba(111,133,84,.9)]" strokeWidth={2.5} />
                </div>
              )}
            </div>
          ))}
        </div>
      </nav>
    </div>
  );
}
