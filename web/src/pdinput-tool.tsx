import { useEffect, useRef, useState } from "preact/hooks";
import { fetchCatalog, fetchDexJson, fetchI18nJa } from "./data";
import { locale, type Locale } from "./i18n";
import {
  buildCookieInstallerScript,
  buildSinglePostSlotFields,
  cookieHeaderBytes,
  encodeTeamForPdinput,
  PDINPUT_COOKIE_LIMIT_BYTES,
  PDINPUT_MAX_BATCH_SIZE,
  PDINPUT_URL,
  splitIntoPdinputBatches,
  type EncodedTeam,
  type PdinputDex,
  type PdinputI18nJa,
} from "./pdinput";
import type { MetaPool } from "./types";

interface CopyStrings {
  eyebrow: string;
  heading: string;
  intro: string;
  limitNote: string;
  batchSelectLabel: string;
  batchPartLabel: (idx: number, start: number, end: number, count: number, bytes: number) => string;
  customSelectLabel: (count: number, bytes: number) => string;
  autoImportSection: string;
  autoImportHelp: string;
  startAutoImport: (count: number) => string;
  stopAutoImport: string;
  openPdinput: string;
  importingStep: (cur: number, total: number, id: string) => string;
  importDone: (total: number) => string;
  popupBlocked: string;
  bookmarkletSection: string;
  bookmarkletHelp: string;
  copyScript: string;
  copiedScript: string;
  bookmarkletLinkLabel: (partLabel: string) => string;
  singleImportBtn: (slot: number) => string;
  singleReceptBtn: string;
  tableHeading: (count: number) => string;
  selectAllInBatch: string;
  clearSelection: string;
  backToGame: string;
  stepIntervalLabel: string;
}

const COPY: Record<Locale, CopyStrings> = {
  ja: {
    eyebrow: "psense.lib.net / PDINPUT.cgi 連携",
    heading: "パーティプール自動インポーター",
    intro:
      "標準パーティプール（team-pool-v2 全33構築）を psense.lib.net/_/PDINPUT.cgi の Cookie 形式（26フィールド PD レコード）に変換し、ブラウザへ一括または個別に自動登録します。",
    limitNote:
      `※ psense.lib.net（Apache 1.3.39）の Cookie ヘッダ上限（8,190 bytes）により、1度に保持できるのは最大 ${PDINPUT_MAX_BATCH_SIZE} パーティ（約7.7KB）です。18パーティ以上登録すると 400 Bad Request になるため、全33パーティは前半（1〜17）と後半（18〜33）、または任意の最大17パーティに分けて取り込みます。`,
    batchSelectLabel: "取り込むバッチ（最大17パーティ）",
    batchPartLabel: (idx, start, end, count, bytes) =>
      `パート ${idx}（#${start}〜#${end} · ${count}構築 · ${bytes.toLocaleString()} B）`,
    customSelectLabel: (count, bytes) =>
      `選択中: ${count} / ${PDINPUT_MAX_BATCH_SIZE} 構築（Cookie サイズ: ${bytes.toLocaleString()} / ${PDINPUT_COOKIE_LIMIT_BYTES.toLocaleString()} B）`,
    autoImportSection: "1. フォーム自動送信でブラウザに取り込む（推奨・ブックマークレット不要）",
    autoImportHelp:
      "「全自動取り込みを開始」を押すと、送信用サブウィンドウを開き、PDINPUT.cgi の 1-POST 保存エンドポイント（ACTION=NEW & ACTION2=SAVED & PBAN=1..N & EUC-JP）へ順番にフォームを自動送信してブラウザの Cookie（PD0〜PD16）へ書き込みます。",
    startAutoImport: (count) => `選択中の ${count} パーティを全自動取り込み`,
    stopAutoImport: "自動取り込みを停止",
    openPdinput: "PDINPUT.cgi を開いて確認",
    importingStep: (cur, total, id) =>
      `取り込み中 (${cur} / ${total}): ${id} を PD${cur - 1} (No.${cur}) に書き込み中…`,
    importDone: (total) =>
      `完了！ ${total} パーティを PD0〜PD${total - 1} に書き込みました。開いたタブ（PDINPUT.cgi）で登録結果を確認してください。`,
    popupBlocked:
      "ポップアップがブロックされました。ブラウザのポップアップ許可を有効にして再度お試しください。",
    bookmarkletSection: "2. ワンライナー / ブックマークレットで一括反映（1秒で完了）",
    bookmarkletHelp:
      "すでに PDINPUT.cgi を開いているタブのアドレスバーやコンソール、またはブックマークレットとして実行すると、古いスロットをクリアして選択中の全パーティを瞬時に Cookie へ書き込みます。",
    copyScript: "ワンライナー (javascript:...) をコピー",
    copiedScript: "コピーしました！",
    bookmarkletLinkLabel: (partLabel) => `PDINPUT一括登録 (${partLabel})`,
    singleImportBtn: (slot) => `No.${slot} に上書き送信`,
    singleReceptBtn: "空き枠へ再現送信 (RECEPT)",
    tableHeading: (count) => `パーティ一覧（${count} 構築）`,
    selectAllInBatch: "このパートの構築を選択",
    clearSelection: "選択をクリア",
    backToGame: "← 対戦画面へ戻る",
    stepIntervalLabel: "送信間隔 (ms)",
  },
  en: {
    eyebrow: "psense.lib.net / PDINPUT.cgi Integration",
    heading: "Automated Party Pool Importer",
    intro:
      "Encodes the built-in party pool (team-pool-v2, all 33 teams) into psense.lib.net/_/PDINPUT.cgi's 26-field PD cookie format and automatically imports them into your browser.",
    limitNote:
      `Note: Because psense.lib.net runs Apache 1.3.39 with an 8,190-byte Cookie header limit, at most ${PDINPUT_MAX_BATCH_SIZE} parties (~7.7 KB) can be stored at once before triggering 400 Bad Request. The 33-team catalog is split into Part 1 (1–17) and Part 2 (18–33), or any custom selection up to 17 teams.`,
    batchSelectLabel: "Batch preset (max 17 teams)",
    batchPartLabel: (idx, start, end, count, bytes) =>
      `Part ${idx} (#${start}–#${end} · ${count} teams · ${bytes.toLocaleString()} B)`,
    customSelectLabel: (count, bytes) =>
      `Selected: ${count} / ${PDINPUT_MAX_BATCH_SIZE} teams (Cookie size: ${bytes.toLocaleString()} / ${PDINPUT_COOKIE_LIMIT_BYTES.toLocaleString()} B)`,
    autoImportSection: "1. Automated Cross-Origin Form Importer (No Bookmarklet Needed)",
    autoImportHelp:
      "Opens a target window and automatically submits EUC-JP forms to PDINPUT.cgi's single-POST slot endpoint (ACTION=NEW & ACTION2=SAVED & PBAN=1..N) to write each team into your browser's PD0..PD16 cookies.",
    startAutoImport: (count) => `Auto-import ${count} selected teams`,
    stopAutoImport: "Stop auto-import",
    openPdinput: "Open PDINPUT.cgi",
    importingStep: (cur, total, id) =>
      `Importing (${cur} / ${total}): writing ${id} to PD${cur - 1} (No.${cur})…`,
    importDone: (total) =>
      `Done! Wrote ${total} teams to PD0–PD${total - 1}. Check the opened PDINPUT.cgi tab.`,
    popupBlocked:
      "Popup was blocked by the browser. Please allow popups for this site and try again.",
    bookmarkletSection: "2. Instant Cookie Installer (Bookmarklet / URL Bar One-Liner)",
    bookmarkletHelp:
      "Drag the bookmarklet to your bookmarks bar or copy the javascript: one-liner and run it on psense.lib.net/_/PDINPUT.cgi to install all selected teams in 1 second.",
    copyScript: "Copy javascript: one-liner",
    copiedScript: "Copied!",
    bookmarkletLinkLabel: (partLabel) => `Install to PDINPUT (${partLabel})`,
    singleImportBtn: (slot) => `Write to No.${slot}`,
    singleReceptBtn: "Append via RECEPT",
    tableHeading: (count) => `Party Pool (${count} teams)`,
    selectAllInBatch: "Select this preset",
    clearSelection: "Clear selection",
    backToGame: "← Back to NC2000",
    stepIntervalLabel: "Step interval (ms)",
  },
};

/** Submit an EUC-JP POST form to `http://psense.lib.net/_/PDINPUT.cgi` targeting `targetWindowName`. */
function submitEucJpForm(
  fields: Record<string, string>,
  targetWindowName: string,
): void {
  const form = document.createElement("form");
  form.method = "POST";
  form.action = PDINPUT_URL;
  form.acceptCharset = "EUC-JP";
  form.target = targetWindowName;
  form.style.display = "none";
  for (const [k, v] of Object.entries(fields)) {
    const input = document.createElement("input");
    input.type = "hidden";
    input.name = k;
    input.value = v;
    form.appendChild(input);
  }
  document.body.appendChild(form);
  form.submit();
  document.body.removeChild(form);
}

export function PdinputTool() {
  const [status, setStatus] = useState<"loading" | "error" | "ready">("loading");
  const [error, setError] = useState("");
  const [allTeams, setAllTeams] = useState<EncodedTeam[]>([]);
  const [batches, setBatches] = useState<EncodedTeam[][]>([]);
  const [activeBatchIdx, setActiveBatchIdx] = useState(0);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [stepMs, setStepMs] = useState(900);
  const [runState, setRunState] = useState<
    | { kind: "idle" }
    | { kind: "running"; current: number; total: number; teamId: string }
    | { kind: "done"; total: number }
    | { kind: "error"; message: string }
  >({ kind: "idle" });
  const [copied, setCopied] = useState(false);
  const timerRef = useRef<number | null>(null);
  const winRef = useRef<Window | null>(null);

  const t = COPY[locale()];

  useEffect(() => {
    void (async () => {
      try {
        const [catalogText, dexRaw, jaRaw] = await Promise.all([
          fetchCatalog(),
          fetchDexJson(),
          fetchI18nJa(),
        ]);
        const catalog = JSON.parse(catalogText) as MetaPool;
        const dex = dexRaw as PdinputDex;
        const ja = jaRaw as PdinputI18nJa;
        const encoded = catalog.teams.map((team) =>
          encodeTeamForPdinput(team, dex, ja),
        );
        const split = splitIntoPdinputBatches(encoded);
        setAllTeams(encoded);
        setBatches(split);
        if (split.length > 0) {
          setSelectedIds(split[0].map((x) => x.id));
        }
        setStatus("ready");
      } catch (e) {
        setError(String(e));
        setStatus("error");
      }
    })();
    return () => {
      if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    };
  }, []);

  if (status === "loading") {
    return (
      <div class="center-screen">
        <div class="loading-pulse">Loading PDINPUT importer…</div>
      </div>
    );
  }
  if (status === "error") {
    return (
      <div class="center-screen">
        <div class="error-box">
          <strong>Error</strong>
          <div>{error}</div>
        </div>
      </div>
    );
  }

  const selectedTeams = selectedIds
    .map((id) => allTeams.find((x) => x.id === id))
    .filter((x): x is EncodedTeam => x !== undefined);
  const currentBytes = cookieHeaderBytes(selectedTeams);
  const installerScript = buildCookieInstallerScript(selectedTeams);

  function selectPreset(idx: number) {
    stopAutoImport();
    setActiveBatchIdx(idx);
    setSelectedIds((batches[idx] ?? []).map((x) => x.id));
  }

  function toggleTeam(id: string) {
    stopAutoImport();
    setSelectedIds((prev) => {
      if (prev.includes(id)) return prev.filter((x) => x !== id);
      if (prev.length >= PDINPUT_MAX_BATCH_SIZE) return prev;
      const nextIds = [...prev, id];
      const nextTeams = nextIds
        .map((tid) => allTeams.find((x) => x.id === tid))
        .filter((x): x is EncodedTeam => x !== undefined);
      if (cookieHeaderBytes(nextTeams) > PDINPUT_COOKIE_LIMIT_BYTES) return prev;
      return nextIds;
    });
  }

  function stopAutoImport() {
    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    setRunState((prev) => (prev.kind === "running" ? { kind: "idle" } : prev));
  }

  function startAutoImport() {
    if (selectedTeams.length === 0) return;
    stopAutoImport();

    const targetName = "nc2000_pdinput_target";
    const w = window.open("about:blank", targetName);
    if (!w) {
      setRunState({ kind: "error", message: t.popupBlocked });
      return;
    }
    winRef.current = w;

    const queue = [...selectedTeams];
    const total = queue.length;

    const runStep = (idx: number) => {
      if (idx >= total) {
        // Navigate the target window to the PDINPUT.cgi top page so the user immediately sees all imported parties.
        try {
          if (winRef.current && !winRef.current.closed) {
            winRef.current.location.href = PDINPUT_URL;
          } else {
            window.open(PDINPUT_URL, targetName);
          }
        } catch {
          window.open(PDINPUT_URL, targetName);
        }
        setRunState({ kind: "done", total });
        timerRef.current = null;
        return;
      }
      const team = queue[idx];
      const pban = idx + 1;
      setRunState({
        kind: "running",
        current: pban,
        total,
        teamId: team.id,
      });
      submitEucJpForm(buildSinglePostSlotFields(team, pban), targetName);
      timerRef.current = window.setTimeout(() => runStep(idx + 1), stepMs);
    };

    runStep(0);
  }

  function sendSingleSlot(team: EncodedTeam, slotNumber: number) {
    const targetName = "nc2000_pdinput_target";
    window.open("about:blank", targetName);
    submitEucJpForm(buildSinglePostSlotFields(team, slotNumber), targetName);
  }

  function sendSingleRecept(team: EncodedTeam) {
    const targetName = "nc2000_pdinput_target";
    window.open("about:blank", targetName);
    submitEucJpForm(
      { ACTION: "RECEPT", C: team.cookieValue },
      targetName,
    );
  }

  async function copyOneLiner() {
    try {
      await navigator.clipboard.writeText(installerScript);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      /* fallback textarea is selectable */
    }
  }

  return (
    <main class="evaluate-page" data-testid="pdinput-page">
      <header class="evaluate-header">
        <div class="evaluate-eyebrow">{t.eyebrow}</div>
        <h1>{t.heading}</h1>
        <p>{t.intro}</p>
        <p class="modal-note" data-testid="pdinput-limit-note">
          {t.limitNote}
        </p>
        <p>
          <a href={import.meta.env.BASE_URL}>{t.backToGame}</a>
          {" · "}
          <a href={PDINPUT_URL} target="_blank" rel="noopener noreferrer">
            {t.openPdinput} ↗
          </a>
        </p>
      </header>

      <section class="evaluate-card" aria-label={t.batchSelectLabel}>
        <h2>{t.batchSelectLabel}</h2>
        <div class="draw-choice" role="group" aria-label={t.batchSelectLabel}>
          {batches.map((b, idx) => {
            const startIdx =
              batches.slice(0, idx).reduce((acc, x) => acc + x.length, 0) + 1;
            const endIdx = startIdx + b.length - 1;
            const bBytes = cookieHeaderBytes(b);
            return (
              <button
                key={idx}
                type="button"
                class="draw-choice-btn"
                data-testid={`pdinput-batch-${idx + 1}`}
                aria-pressed={activeBatchIdx === idx}
                onClick={() => selectPreset(idx)}
              >
                {t.batchPartLabel(idx + 1, startIdx, endIdx, b.length, bBytes)}
              </button>
            );
          })}
        </div>
        <p class="modal-note" data-testid="pdinput-selection-summary">
          <strong>{t.customSelectLabel(selectedTeams.length, currentBytes)}</strong>
        </p>
      </section>

      <section class="evaluate-card">
        <h2>{t.autoImportSection}</h2>
        <p class="modal-note">{t.autoImportHelp}</p>
        <div class="pool-actions" style={{ alignItems: "center", gap: "12px", flexWrap: "wrap" }}>
          <button
            type="button"
            class="primary"
            data-testid="pdinput-auto-start"
            disabled={selectedTeams.length === 0 || runState.kind === "running"}
            onClick={startAutoImport}
          >
            {t.startAutoImport(selectedTeams.length)}
          </button>
          {runState.kind === "running" && (
            <button
              type="button"
              class="ghost"
              data-testid="pdinput-auto-stop"
              onClick={stopAutoImport}
            >
              {t.stopAutoImport}
            </button>
          )}
          <label style={{ display: "inline-flex", alignItems: "center", gap: "6px", fontSize: "0.9rem" }}>
            <span>{t.stepIntervalLabel}:</span>
            <input
              type="number"
              min={300}
              max={5000}
              step={100}
              value={stepMs}
              style={{ width: "80px" }}
              onChange={(e) =>
                setStepMs(Math.max(300, Number((e.target as HTMLInputElement).value) || 900))
              }
            />
          </label>
        </div>
        {runState.kind !== "idle" && (
          <div
            class="pool-report"
            data-testid="pdinput-status"
            role="status"
            style={{ marginTop: "12px" }}
          >
            {runState.kind === "running" && (
              <div>
                {t.importingStep(runState.current, runState.total, runState.teamId)}
              </div>
            )}
            {runState.kind === "done" && (
              <div class="pool-verdict ok">{t.importDone(runState.total)}</div>
            )}
            {runState.kind === "error" && (
              <div class="pool-verdict no">{runState.message}</div>
            )}
          </div>
        )}
      </section>

      <section class="evaluate-card">
        <h2>{t.bookmarkletSection}</h2>
        <p class="modal-note">{t.bookmarkletHelp}</p>
        <div class="pool-actions" style={{ gap: "12px", flexWrap: "wrap", marginBottom: "8px" }}>
          <button
            type="button"
            class="ghost"
            data-testid="pdinput-copy-script"
            disabled={selectedTeams.length === 0}
            onClick={() => void copyOneLiner()}
          >
            {copied ? t.copiedScript : t.copyScript}
          </button>
          <a
            href={installerScript}
            class="ghost"
            data-testid="pdinput-bookmarklet"
            style={{
              display: "inline-block",
              padding: "6px 12px",
              border: "1px solid currentColor",
              borderRadius: "6px",
              textDecoration: "none",
            }}
            onClick={(e) => e.preventDefault()}
          >
            {t.bookmarkletLinkLabel(`Part ${activeBatchIdx + 1}`)}
          </a>
        </div>
        <textarea
          readOnly
          rows={3}
          data-testid="pdinput-script-text"
          value={installerScript}
          style={{ width: "100%", fontFamily: "monospace", fontSize: "0.8rem" }}
          onClick={(e) => (e.currentTarget as HTMLTextAreaElement).select()}
        />
      </section>

      <section class="evaluate-card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "8px" }}>
          <h2>{t.tableHeading(allTeams.length)}</h2>
          <div style={{ display: "flex", gap: "8px" }}>
            <button
              type="button"
              class="ghost"
              onClick={() => selectPreset(activeBatchIdx)}
            >
              {t.selectAllInBatch}
            </button>
            <button
              type="button"
              class="ghost"
              onClick={() => {
                stopAutoImport();
                setSelectedIds([]);
              }}
            >
              {t.clearSelection}
            </button>
          </div>
        </div>
        <div style={{ display: "grid", gap: "10px", marginTop: "12px" }}>
          {allTeams.map((team, idx) => {
            const slotIdx = selectedIds.indexOf(team.id);
            const checked = slotIdx !== -1;
            const targetSlot = checked ? slotIdx + 1 : (idx % PDINPUT_MAX_BATCH_SIZE) + 1;
            return (
              <div
                key={team.id}
                data-testid={`pdinput-team-${team.id}`}
                style={{
                  border: "1px solid rgba(128,128,128,0.3)",
                  borderRadius: "8px",
                  padding: "10px 12px",
                  display: "flex",
                  flexDirection: "column",
                  gap: "6px",
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "8px" }}>
                  <label style={{ display: "inline-flex", alignItems: "center", gap: "8px", fontWeight: 600, cursor: "pointer" }}>
                    <input
                      type="checkbox"
                      checked={checked}
                      disabled={!checked && selectedIds.length >= PDINPUT_MAX_BATCH_SIZE}
                      onChange={() => toggleTeam(team.id)}
                    />
                    <span>
                      #{idx + 1} {team.id}
                    </span>
                    {checked && (
                      <span class="badge" data-testid={`pdinput-slot-badge-${team.id}`}>
                        → PD{slotIdx} (No.{slotIdx + 1})
                      </span>
                    )}
                  </label>
                  <div style={{ display: "flex", gap: "6px", flexWrap: "wrap" }}>
                    <button
                      type="button"
                      class="ghost"
                      data-testid={`pdinput-write-${team.id}`}
                      onClick={() => sendSingleSlot(team, targetSlot)}
                    >
                      {t.singleImportBtn(targetSlot)}
                    </button>
                    <button
                      type="button"
                      class="ghost"
                      data-testid={`pdinput-recept-${team.id}`}
                      onClick={() => sendSingleRecept(team)}
                    >
                      {t.singleReceptBtn}
                    </button>
                  </div>
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: "8px", fontSize: "0.88rem" }}>
                  {team.mons.map((m, mIdx) => (
                    <span
                      key={mIdx}
                      style={{
                        background: "rgba(128,128,128,0.12)",
                        padding: "2px 8px",
                        borderRadius: "4px",
                      }}
                    >
                      {m.jpSpecies} Lv{m.level} ({m.jpItem || (locale() === "ja" ? "なし" : "None")}) [{m.jpMoves.filter(Boolean).join(" / ")}]
                    </span>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </section>
    </main>
  );
}
