import type { Metadata } from "next";
import "./globals.css";
import { BankProvider } from "@/lib/bank-context";
import { FeaturesProvider } from "@/lib/features-context";
import { ThemeProvider } from "@/lib/theme-context";
import { Toaster } from "@/components/ui/sonner";

export const metadata: Metadata = {
  title: "HMS Control Plane",
  description: "Control plane for the temporal semantic memory system",
  icons: {
    icon: "/favicon.png",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="bg-background text-foreground">
        <ThemeProvider>
          <FeaturesProvider>
            <BankProvider>{children}</BankProvider>
          </FeaturesProvider>
        </ThemeProvider>
        <Toaster />
      </body>
    </html>
  );
}
