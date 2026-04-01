import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Autoresearch Progress Watcher",
  description: "Autoresearch Progress Watcher",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        {children}
      </body>
    </html>
  );
}