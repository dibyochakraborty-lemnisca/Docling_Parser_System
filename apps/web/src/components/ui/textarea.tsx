import * as React from "react";
import { cn } from "@/lib/utils";

export interface TextareaProps
  extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {}

const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, ...props }, ref) => (
    <textarea
      className={cn(
        // Lemnisca input well: surface-1, hairline, light text, teal
        // caret + glow on focus. (text-sm is twMerge-known so the color
        // class is never collapsed away — keeps text visible on dark.)
        "flex min-h-[80px] w-full rounded-md border border-rule bg-surface-1 px-3 py-2 text-sm text-ink caret-accent placeholder:text-ink-faint transition-colors focus-visible:border-accent-deep focus-visible:outline-none focus-visible:shadow-glow-soft disabled:cursor-not-allowed disabled:opacity-50",
        className,
      )}
      ref={ref}
      {...props}
    />
  ),
);
Textarea.displayName = "Textarea";

export { Textarea };
