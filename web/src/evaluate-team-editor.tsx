import { toolText } from "./tool-strings";
import { useId, useMemo, useRef, useState } from "preact/hooks";
import { locale, itemName, moveName, speciesName } from "./i18n";
import { parsePsExport } from "./ps-import";

export interface EditorDex {
  species: Record<string, { name: string }>;
  moves: Record<string, { name: string }>;
  items: Record<string, { name: string }>;
}
interface EditorSet {
  species: string;
  level?: number;
  item?: string;
  moves?: string[];
  happiness?: number;
  ivs?: Record<string, number>;
  evs?: Record<string, number>;
  [key: string]: unknown;
}
interface Choice {
  value: string;
  label: string;
}
const normalize = (text: string) =>
  text
    .normalize("NFKC")
    .toLowerCase()
    .replace(/[ぁ-ゖ]/g, (char) =>
      String.fromCharCode(char.charCodeAt(0) + 0x60),
    );
const newMember = (): EditorSet => ({
  species: "",
  level: 50,
  item: "",
  moves: [],
});

function parseEditor(text: string): EditorSet[] | null {
  if (!text.trim()) return [newMember()];
  try {
    const parsed = /^\s*[\[{]/.test(text)
      ? JSON.parse(text)
      : parsePsExport(text);
    if (parsed.findings?.length) return null;
    const sets = Array.isArray(parsed) ? parsed : parsed.sets;
    if (
      !Array.isArray(sets) ||
      !sets.length ||
      sets.length > 6 ||
      sets.some(
        (s) =>
          !s ||
          typeof s !== "object" ||
          typeof s.species !== "string" ||
          (s.moves !== undefined &&
            (!Array.isArray(s.moves) ||
              s.moves.some((m: unknown) => typeof m !== "string") ||
              s.moves.length > 4)),
      )
    )
      return null;
    return sets;
  } catch {
    return null;
  }
}

function NameChoice({
  label,
  caption,
  value,
  choices,
  onChange,
  placeholder,
}: {
  label: string;
  caption: string;
  value: string;
  choices: Choice[];
  onChange: (value: string) => void;
  placeholder: string;
}) {
  const id = useId();
  const box = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  const [highlight, setHighlight] = useState(0);
  const [query, setQuery] = useState<string | null>(null);
  const display =
    choices.find((c) => normalize(c.value) === normalize(value))?.label ??
    value;
  const text = query ?? display;
  const matches = useMemo(() => {
    const term = normalize(text);
    return choices
      .filter(
        (c) =>
          normalize(c.label).includes(term) ||
          normalize(c.value).includes(term),
      )
      .slice(0, 8);
  }, [text, choices]);
  function pick(choice: Choice) {
    onChange(choice.value);
    setQuery(null);
    setOpen(false);
    setHighlight(0);
  }
  return (
    <div
      class="eval-name-choice"
      ref={box}
      onBlurCapture={(e) => {
        if (!box.current?.contains(e.relatedTarget as Node | null)) {
          setOpen(false);
          setQuery(null);
        }
      }}
    >
      <label htmlFor={id}>{caption}</label>
      <input
        id={id}
        aria-label={label}
        role="combobox"
        aria-autocomplete="list"
        aria-expanded={open}
        aria-controls={`${id}-options`}
        aria-activedescendant={
          open && matches[highlight] ? `${id}-${highlight}` : undefined
        }
        autoComplete="off"
        placeholder={placeholder}
        value={text}
        onFocus={() => {
          setOpen(true);
          setHighlight(0);
        }}
        onInput={(e) => {
          const entered = e.currentTarget.value;
          setQuery(entered);
          setOpen(true);
          setHighlight(0);
          const exact = choices.find(
            (c) =>
              normalize(c.label) === normalize(entered) ||
              normalize(c.value) === normalize(entered),
          );
          onChange(exact?.value ?? entered);
        }}
        onKeyDown={(e) => {
          if (e.isComposing) return;
          if (e.key === "Escape") {
            setOpen(false);
            setQuery(null);
          }
          if (
            (e.key === "ArrowDown" || e.key === "ArrowUp") &&
            matches.length
          ) {
            e.preventDefault();
            setOpen(true);
            setHighlight(
              (n) =>
                (n + (e.key === "ArrowDown" ? 1 : matches.length - 1)) %
                matches.length,
            );
          }
          if (e.key === "Enter" && open && matches[highlight]) {
            e.preventDefault();
            pick(matches[highlight]);
          }
        }}
      />
      {open && (
        <div
          class="eval-name-options"
          id={`${id}-options`}
          role="listbox"
          aria-label={toolText("suggestionsFor", label)}
        >
          {matches.map((c, i) => (
            <button
              key={c.value}
              id={`${id}-${i}`}
              type="button"
              role="option"
              aria-selected={i === highlight}
              tabIndex={-1}
              onPointerDown={(e) => e.preventDefault()}
              onClick={() => pick(c)}
            >
              {c.label}
            </button>
          ))}
          {!matches.length && <p>{toolText("noSuggestions")}</p>}
        </div>
      )}
    </div>
  );
}

export function TeamEditor({
  text,
  onChange,
  dex,
  name,
}: {
  text: string;
  onChange: (text: string) => void;
  dex: EditorDex;
  name: string;
}) {
  const [active, setActive] = useState(0);
  const sets = useMemo(() => parseEditor(text), [text]);
  const choices = useMemo(() => {
    const list = (kind: keyof EditorDex, label: (value: string) => string) =>
      Object.values(dex[kind])
        .map((v) => ({ value: v.name, label: label(v.name) }))
        .sort((a, b) => a.label.localeCompare(b.label, locale()));
    return {
      species: list("species", speciesName),
      moves: list("moves", moveName),
      items: [{ value: "", label: toolText("none") }, ...list("items", itemName)],
    };
  }, [dex, locale()]);
  if (!sets)
    return (
      <p class="eval-warning">{toolText("editorInvalid")}</p>
    );
  const index = Math.min(active, sets.length - 1);
  const member = sets[index];
  const key = toolText("memberKey", name, index + 1);
  const replace = (next: EditorSet[]) => onChange(JSON.stringify(next));
  const patch = (updates: Partial<EditorSet>) =>
    replace(sets.map((s, i) => (i === index ? { ...s, ...updates } : s)));
  const moves = member.moves ?? [];
  const dv = (stat: string) => Math.floor((member.ivs?.[stat] ?? 31) / 2);
  function setDv(stat: string, value: number) {
    const ivs = {
      hp: 30,
      atk: 30,
      def: 30,
      spa: 30,
      spd: 30,
      spe: 30,
      ...member.ivs,
      [stat]: value * 2,
    };
    ivs.spd = ivs.spa;
    ivs.hp =
      ((Math.floor(ivs.atk / 2) % 2) * 8 +
        (Math.floor(ivs.def / 2) % 2) * 4 +
        (Math.floor(ivs.spe / 2) % 2) * 2 +
        (Math.floor(ivs.spa / 2) % 2)) *
      2;
    patch({ ivs });
  }
  return (
    <div class="eval-team-editor">
      <div class="eval-members" aria-label={toolText("teamMembers", name)}>
        {sets.map((s, i) => (
          <button
            type="button"
            key={i}
            class={index === i ? "selected" : ""}
            aria-pressed={index === i}
            onClick={() => setActive(i)}
          >
            {i + 1}. {s.species ? speciesName(s.species) : toolText("emptyMember")}
          </button>
        ))}
        {sets.length < 6 && (
          <button
            type="button"
            onClick={() => {
              setActive(sets.length);
              replace([...sets, newMember()]);
            }}
          >{toolText("addPokemon")}</button>
        )}
      </div>
      <div class="eval-member-form" key={key}>
        <NameChoice
          label={toolText("memberSpecies", key)}
          caption={toolText("pokemon")}
          value={member.species}
          choices={choices.species}
          placeholder={toolText("speciesExample")}
          onChange={(species) =>
            patch({
              species,
              ...(member.name === member.species ? { name: species } : {}),
            })
          }
        />
        <div class="eval-member-basics">
          <label>
            {toolText("level")}<input
              aria-label={toolText("memberLevel", key)}
              type="number"
              min="1"
              max="100"
              value={member.level ?? 55}
              onInput={(e) =>
                patch({
                  level: e.currentTarget.value
                    ? Number(e.currentTarget.value)
                    : 0,
                })
              }
            />
          </label>
          <NameChoice
            label={toolText("memberItem", key)}
            caption={toolText("item")}
            value={member.item ?? ""}
            choices={choices.items}
            placeholder={toolText("itemPlaceholder")}
            onChange={(item) => patch({ item })}
          />
        </div>
        <div class="eval-moves">
          {[0, 1, 2, 3].map((slot) => (
            <NameChoice
              key={slot}
              label={toolText("memberMove", key, slot + 1)}
              caption={toolText("moveSlot", slot + 1)}
              value={moves[slot] ?? ""}
              choices={choices.moves}
              placeholder={toolText("movePlaceholder")}
              onChange={(value) => {
                const next = [...moves];
                while (next.length < 4) next.push("");
                next[slot] = value;
                patch({ moves: next });
              }}
            />
          ))}
        </div>
        <details>
          <summary>{toolText("advancedStats")}</summary>
          <p class="eval-muted">{toolText("advancedStatsHelp")}</p>
          <div class="eval-detail-fields">
            {[
              ["atk", toolText("attack")],
              ["def", toolText("defense")],
              ["spe", toolText("speed")],
              ["spa", toolText("special")],
            ].map(([stat, label]) => (
              <label key={stat}>
                {label}{toolText("dvSuffix")}<input
                  aria-label={toolText("memberDv", key, label)}
                  type="number"
                  min="0"
                  max="15"
                  value={dv(stat)}
                  onInput={(e) => setDv(stat, Number(e.currentTarget.value))}
                />
              </label>
            ))}
            <label>
              {toolText("happiness")}<input
                aria-label={toolText("memberHappiness", key)}
                type="number"
                min="0"
                max="255"
                value={member.happiness ?? 255}
                onInput={(e) =>
                  patch({ happiness: Number(e.currentTarget.value) })
                }
              />
            </label>
            <label>
              {toolText("training")}<select
                aria-label={toolText("memberTraining", key)}
                value={
                  !member.evs ||
                  Object.values(member.evs).every((v) => v === 255)
                    ? "max"
                    : Object.values(member.evs).every((v) => v === 0)
                      ? "none"
                      : "custom"
                }
                onChange={(e) => {
                  const v = e.currentTarget.value === "max" ? 255 : 0;
                  patch({
                    evs: { hp: v, atk: v, def: v, spa: v, spd: v, spe: v },
                  });
                }}
              >
                <option value="max">{toolText("trainingMax")}</option>
                <option value="none">{toolText("trainingNone")}</option>
                <option value="custom" disabled>{toolText("trainingCustom")}</option>
              </select>
            </label>
          </div>
        </details>
        {sets.length > 1 && (
          <button
            type="button"
            class="ghost"
            onClick={() => {
              replace(sets.filter((_, i) => i !== index));
              setActive(Math.max(0, index - 1));
            }}
          >{toolText("removePokemon")}</button>
        )}
      </div>
    </div>
  );
}
