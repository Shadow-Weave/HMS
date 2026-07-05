import { Tag } from "lucide-react";

interface TagListProps {
  tags: string[];
  /** Visual size variant. Defaults to "sm". */
  size?: "xs" | "sm" | "md";
  /** Show the "Tags" section label above the chips. Defaults to false. */
  showLabel?: boolean;
}

/**
 * Renders a list of tags as branded chips using HMS's primary color.
 * Returns null when the tags array is empty or undefined.
 */
export function TagList({ tags, size = "sm", showLabel = false }: TagListProps) {
  if (!tags || tags.length === 0) return null;

  const chipClass =
    size === "xs"
      ? "text-[10px] px-1.5 py-0.5 rounded gap-0.5"
      : size === "md"
        ? "text-sm px-3 py-1 rounded-full gap-1 font-medium"
        : "text-xs px-2 py-0.5 rounded-md gap-1 font-medium";

  return (
    <div>
      {showLabel && (
        <div className="text-xs font-bold text-muted-foreground uppercase mb-2 flex items-center gap-1">
          <Tag className="w-3 h-3" />
          Tags
        </div>
      )}
      <div className="flex flex-wrap gap-1.5">
        {tags.map((tag, idx) => (
          <span
            key={idx}
            className={`inline-flex items-center ${chipClass} bg-primary/10 text-primary border border-primary/20 leading-none`}
          >
            <span className="opacity-50 select-none font-mono">#</span>
            {tag}
          </span>
        ))}
      </div>
    </div>
  );
}
