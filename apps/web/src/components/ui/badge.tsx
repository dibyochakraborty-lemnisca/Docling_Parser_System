import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

// Lemnisca status pills: mono, uppercase, wide-tracked. Tinted fill +
// colored hairline + colored text — no solid chips. Teal is the
// signature (default); ok/warn/error carry semantic meaning only.
const badgeVariants = cva(
  "inline-flex items-center rounded-full border px-2.5 py-0.5 font-ui text-ui-xs font-medium uppercase tracking-[0.1em] transition-colors",
  {
    variants: {
      variant: {
        default:
          "border-accent/40 bg-accent-soft text-accent",
        secondary:
          "border-rule bg-surface-2 text-ink-muted",
        destructive:
          "border-error/40 bg-error/10 text-error",
        outline: "border-rule text-ink-faint",
        success:
          "border-ok/40 bg-ok/10 text-ok",
        warning:
          "border-warn/40 bg-warn/10 text-warn",
      },
    },
    defaultVariants: { variant: "default" },
  },
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />;
}

export { Badge, badgeVariants };
