import { useToolMessage } from "./tool-message";
import { toolText } from "./tool-strings";
import { useEffect, useState } from "preact/hooks";
import type { Battle } from "./engine";
import type { InfoMode } from "./info-mode";
import type { StateView } from "./types";
import { copyKifu, kifuUrl } from "./kifu-code";
import "./kifu.css";

export function KifuExport(props: { battle: Battle; mode: InfoMode; state: StateView }) {
  const [code, setCode] = useState("");
  const [notice, setNotice] = useToolMessage("");
  const [manual, setManual] = useState(false);
  useEffect(() => {
    setManual(false);
    setNotice("");
    try {
      setCode(props.battle.exportKifu(props.mode === "open", 1));
    } catch (error) {
      console.error("kifu export", error);
      setCode("");
      setNotice(() => toolText("exportReplayError"));
    }
  }, [props.battle, props.mode, props.state]);

  async function copy() {
    const copied = await copyKifu(code);
    setManual(!copied);
    setNotice(() => copied ? toolText("replayCopied") : toolText("copyReplayFallback"));
  }

  return <section class="kifu-tool kp-export-shell" aria-labelledby="kifu-export-title">
    <div class="kp-card">
      <h2 id="kifu-export-title">{toolText("saveBattle")}</h2>
      <p class="kp-muted">{toolText("saveBattleHelp")}</p>
      <div class="kp-actions">
        <button class="primary" disabled={!code} onClick={() => void copy()}>{toolText("copyReplay")}</button>
        {code && <a class="kp-button" href={kifuUrl(code)} target="_blank" rel="noopener">{toolText("tryAnotherMove")}<span class="sr-only">{toolText("newTab")}</span></a>}
      </div>
      <p class="kp-status" role="status">{notice}</p>
      {manual && <div class="kp-copy-fallback">
        <label for="kifu-copy-code">{toolText("replayToCopy")}</label>
        <textarea id="kifu-copy-code" readOnly value={code} rows={4} onFocus={e => e.currentTarget.select()} />
        <p class="kp-muted">{toolText("manualCopyHelp")}</p>
      </div>}
    </div>
  </section>;
}
