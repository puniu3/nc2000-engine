import { useLayoutEffect, useState } from "preact/hooks";
import { locale, setLocale, subscribeLocale, ui, type Locale } from "./i18n";
import { toolText } from "./tool-strings";
import { render } from "preact";
import { App } from "./app";
import { readDoor } from "./info-mode";
import { Evaluate } from "./evaluate";
import { Fork } from "./fork";
import { KifuTool } from "./kifu";
import { initTooltips } from "./tooltip";
import { initAnnouncer } from "./announcer";
import "./style.css";

initTooltips();
initAnnouncer();
const door = readDoor();
const params = new URLSearchParams(location.search);
const forkName = params.get("fork")?.trim();
const legacyFork = forkName && !["1", "true"].includes(forkName.toLowerCase());
function Root() {
  const [language, setLanguage] = useState(locale);
  useLayoutEffect(() => subscribeLocale(() => setLanguage(locale())), []);
  const kifu = params.has("kifu-preview") || (door === "fork" && !legacyFork && !params.has("advanced"));
  const tool = kifu || door === "evaluate" || door === "fork";
  useLayoutEffect(() => {
    if (tool) document.title = toolText(kifu ? "kifuTitle" : door === "evaluate" ? "evaluateTitle" : "forkTitle");
  }, [language]);
  return <>
    {tool && <div class="tool-language-bar">
      <select aria-label={ui().languageLabel} value={language} onChange={e => setLocale(e.currentTarget.value as Locale)}>
        <option value="en">English</option>
        <option value="ja">日本語</option>
      </select>
    </div>}
    {kifu ? <KifuTool /> : door === "evaluate" ? <Evaluate /> : door === "fork" ? <Fork /> : <App />}
  </>;
}

render(<Root />, document.getElementById("app")!);
