import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { ArrowUp, Square } from "lucide-react";

interface Props {
  disabled: boolean;
  busy: boolean;
  placeholder: string;
  onSend: (text: string) => void;
  onStop: () => void;
  prefill?: { text: string; nonce: number } | null;
}

export function Composer({ disabled, busy, placeholder, onSend, onStop, prefill }: Props) {
  const [text, setText] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (prefill) {
      setText(prefill.text);
      ref.current?.focus();
    }
  }, [prefill]);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [text]);

  function submit() {
    const t = text.trim();
    if (!t || disabled || busy) return;
    onSend(t);
    setText("");
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
    }
  }

  return (
    <form
      className="composer"
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
    >
      <label htmlFor="question" className="sr-only">
        Ask a question about your data
      </label>
      <textarea
        id="question"
        ref={ref}
        rows={1}
        value={text}
        placeholder={placeholder}
        disabled={disabled}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKeyDown}
      />
      {busy ? (
        <button type="button" className="composer__send is-stop" onClick={onStop} aria-label="Stop answering">
          <Square size={14} aria-hidden />
        </button>
      ) : (
        <button type="submit" className="composer__send" disabled={disabled || !text.trim()} aria-label="Ask">
          <ArrowUp size={17} aria-hidden />
        </button>
      )}
    </form>
  );
}
