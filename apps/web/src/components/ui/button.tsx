import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

// Lemnisca buttons: mono, uppercase, wide-tracked, fully-round pills.
// Primary is the solid teal signal; outline/secondary are hairline
// chrome that lift to teal on hover. Focus uses the teal glow ring.
const buttonVariants = cva(
  "inline-flex items-center justify-center whitespace-nowrap rounded-full font-ui text-ui-xs font-medium uppercase tracking-[0.1em] transition-all focus-visible:outline-none focus-visible:shadow-glow-soft disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default:
          "bg-accent text-accent-ink hover:bg-accent-bright",
        destructive:
          "border border-error/40 bg-error/10 text-error hover:bg-error/20",
        outline:
          "border border-rule bg-transparent text-ink hover:border-accent-deep hover:bg-accent-soft hover:text-accent",
        secondary:
          "border border-rule bg-surface-1 text-ink-secondary hover:border-accent-deep hover:bg-surface-2 hover:text-ink",
        ghost: "text-ink-secondary hover:bg-surface-2 hover:text-ink",
        link: "text-accent underline-offset-4 hover:underline",
      },
      size: {
        default: "h-10 px-5 py-2",
        sm: "h-9 px-4",
        lg: "h-11 px-8",
        icon: "h-10 w-10",
      },
    },
    defaultVariants: { variant: "default", size: "default" },
  },
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    );
  },
);
Button.displayName = "Button";

export { Button, buttonVariants };
