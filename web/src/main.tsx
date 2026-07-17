import { render } from "preact";
import { App } from "./app";
import { initTooltips } from "./tooltip";
import { initAnnouncer } from "./announcer";
import "./style.css";

initTooltips();
initAnnouncer();
render(<App />, document.getElementById("app")!);
