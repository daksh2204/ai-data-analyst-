import { memo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { CodeBlock } from "./CodeBlock";

function MarkdownImpl({ children }: { children: string }) {
  return (
    <div className="prose">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          code({ className, children: code, ...rest }) {
            const match = /language-(\w+)/.exec(className ?? "");
            const text = String(code).replace(/\n$/, "");
            if (match || text.includes("\n")) {
              return <CodeBlock code={text} language={match?.[1] ?? ""} />;
            }
            return (
              <code className="inline-code" {...rest}>
                {code}
              </code>
            );
          },
          pre({ children: inner }) {
            return <>{inner}</>;
          },
          table({ children: inner }) {
            return (
              <div className="prose-table">
                <table>{inner}</table>
              </div>
            );
          },
          a({ children: inner, href }) {
            return (
              <a href={href} target="_blank" rel="noreferrer noopener">
                {inner}
              </a>
            );
          },
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}

export const Markdown = memo(MarkdownImpl);
