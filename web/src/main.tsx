import { render } from "preact";
import { App } from "./app";
import { initTooltips } from "./tooltip";
import "./style.css";

initTooltips();
render(<App />, document.getElementById("app")!);
