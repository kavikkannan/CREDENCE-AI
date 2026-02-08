import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'GenAI Social Media Credibility Analyzer',
  description: 'Research prototype for analyzing social media credibility using multi-agentic pipeline',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
