import { useCallback, useRef, useState } from "react";
import { ApiError, streamChat } from "../api/client";
import type { AssistantMessage, ChatEvent, Step } from "../types";
import type { LiveTurn } from "../components/chat/Turn";

function upsertStep(steps: Step[], step: Step): Step[] {
  const i = steps.findIndex((s) => s.id === step.id);
  if (i === -1) return [...steps, step];
  const next = steps.slice();
  next[i] = { ...next[i], ...step };
  return next;
}

export function reduceLive(turn: LiveTurn, ev: ChatEvent): LiveTurn {
  switch (ev.type) {
    case "start":
      return { ...turn, provider: ev.provider };
    case "token":
      return { ...turn, text: turn.text + ev.text };
    case "thought":
      // Text written before tool calls is part of the working, not the answer.
      return {
        ...turn,
        text: "",
        steps: [...turn.steps, { id: `note-${turn.steps.length}`, kind: "note", text: ev.text.trim() }],
      };
    case "step_start":
    case "step_end":
      return { ...turn, steps: upsertStep(turn.steps, ev.step) };
    case "artifact":
      return turn.artifacts.some((a) => a.id === ev.artifact.id) ? turn : { ...turn, artifacts: [...turn.artifacts, ev.artifact] };
    case "notice":
      return { ...turn, notices: [...turn.notices, ev.message] };
    case "error":
      return { ...turn, error: ev.message };
    default:
      return turn;
  }
}

export function useChat(sessionId: string | null, onFinal: (question: string, message: AssistantMessage) => void) {
  const [live, setLive] = useState<LiveTurn | null>(null);
  const controller = useRef<AbortController | null>(null);

  const ask = useCallback(
    async (question: string) => {
      if (!sessionId) return;
      const ctrl = new AbortController();
      controller.current = ctrl;
      let turn: LiveTurn = { question, text: "", steps: [], artifacts: [], notices: [], error: null };
      let finished = false;
      setLive(turn);
      try {
        await streamChat(
          sessionId,
          question,
          (ev) => {
            if (ev.type === "final") {
              finished = true;
              onFinal(question, ev.message);
              setLive(null);
              return;
            }
            turn = reduceLive(turn, ev);
            setLive(turn);
          },
          ctrl.signal,
        );
        if (!finished && !turn.error) {
          if (ctrl.signal.aborted) setLive({ ...turn, error: "Stopped." });
          else setLive({ ...turn, error: "The connection closed before the answer finished. Try again." });
        }
      } catch (e) {
        setLive({ ...turn, error: e instanceof ApiError ? e.message : "Something went wrong while answering." });
      } finally {
        controller.current = null;
      }
    },
    [sessionId, onFinal],
  );

  const stop = useCallback(() => controller.current?.abort(), []);
  const busy = live !== null && !live.error;
  const clearLive = useCallback(() => setLive(null), []);
  return { live, ask, stop, busy, clearLive };
}
