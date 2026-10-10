import { useEffect, useRef } from "react";
import type { ChatMessage } from "../../types";
import { CompletedTurn, LiveTurnView, type LiveTurn } from "./Turn";

interface Props {
  messages: ChatMessage[];
  live: LiveTurn | null;
  suggestions: string[];
  hasData: boolean;
  offline: boolean;
  onPick: (q: string) => void;
}

export function Conversation({ messages, live, suggestions, hasData, offline, onPick }: Props) {
  const endRef = useRef<HTMLDivElement>(null);
  const assistant = messages.filter((m) => m.role === "assistant");
  const liveSize = live ? live.text.length + live.steps.length * 100 + live.artifacts.length * 1000 : 0;

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [assistant.length, liveSize]);

  if (!assistant.length && !live) {
    return (
      <div className="conversation conversation--empty">
        <div className="starter">
          <h1 className="starter__title">{hasData ? "What do you want to know?" : "Start with a CSV file"}</h1>
          <p className="starter__lede">
            {hasData
              ? "Ask in plain English. Every answer shows the queries behind it, so you can check the numbers."
              : "Upload one or more CSV files, or load the sample sales data, then ask questions in plain English."}
          </p>
          {hasData && suggestions.length > 0 && (
            <ul className="starter__suggestions" aria-label="Suggested questions">
              {suggestions.map((s) => (
                <li key={s}>
                  <button type="button" onClick={() => onPick(s)}>
                    {s}
                  </button>
                </li>
              ))}
            </ul>
          )}
          {hasData && offline && (
            <p className="starter__hint">
              Running on the built-in offline engine. Add an LLM API key to the server for open-ended questions.
            </p>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="conversation">
      {assistant.map((m) => m.role === "assistant" && <CompletedTurn key={m.id} message={m} />)}
      {live && <LiveTurnView turn={live} />}
      <div ref={endRef} />
    </div>
  );
}
