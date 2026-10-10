import { CircleAlert, Info, Zap } from "lucide-react";
import type { AssistantMessage, Artifact, Step } from "../../types";
import { Markdown } from "../common/Markdown";
import { ArtifactView, sortArtifacts } from "./ArtifactView";
import { Working } from "./Working";

export interface LiveTurn {
  question: string;
  text: string;
  steps: Step[];
  artifacts: Artifact[];
  notices: string[];
  error: string | null;
  provider?: string;
}

function providerLabel(provider: string, model: string): string {
  if (provider === "offline") return "Offline engine";
  return `${provider} ${model}`;
}

export function CompletedTurn({ message }: { message: AssistantMessage }) {
  return (
    <article className="turn">
      <h2 className="turn__question">{message.question}</h2>
      <Markdown>{message.answer}</Markdown>
      {message.artifacts.length > 0 && (
        <div className="turn__artifacts">
          {sortArtifacts(message.artifacts).map((a) => (
            <ArtifactView key={a.id} artifact={a} />
          ))}
        </div>
      )}
      <Working steps={message.steps} code={message.code} elapsedMs={message.elapsed_ms} />
      <p className="turn__meta">
        {message.cached && (
          <span className="turn__badge">
            <Zap size={12} aria-hidden /> From cache
          </span>
        )}
        <span>{providerLabel(message.provider, message.model)}</span>
        {message.usage.input_tokens > 0 && (
          <span>
            {(message.usage.input_tokens + message.usage.output_tokens).toLocaleString("en-GB")} tokens
          </span>
        )}
      </p>
    </article>
  );
}

export function LiveTurnView({ turn }: { turn: LiveTurn }) {
  const running = !turn.error;
  return (
    <article className="turn is-live" aria-live="polite" aria-busy={running}>
      <h2 className="turn__question">{turn.question}</h2>
      {turn.notices.map((n, i) => (
        <p key={i} className="notice">
          <Info size={14} aria-hidden /> {n}
        </p>
      ))}
      <Working steps={turn.steps} code={[]} live={running} defaultOpen />
      {turn.artifacts.length > 0 && (
        <div className="turn__artifacts">
          {sortArtifacts(turn.artifacts).map((a) => (
            <ArtifactView key={a.id} artifact={a} />
          ))}
        </div>
      )}
      {turn.text ? (
        <Markdown>{turn.text}</Markdown>
      ) : running ? (
        <p className="thinking" aria-label="Thinking">
          <span />
          <span />
          <span />
        </p>
      ) : null}
      {turn.error && (
        <p className="notice notice--error" role="alert">
          <CircleAlert size={14} aria-hidden /> {turn.error}
        </p>
      )}
    </article>
  );
}
