# GenAI Social Media Credibility Analyzer - Frontend

A Next.js + Tailwind CSS dashboard for visualizing and interacting with the GenAI Social Media Credibility Analyzer pipeline.

## Features

- 🎨 Modern, dark-themed UI with smooth animations
- 📊 Interactive pipeline visualization
- 🔍 Data explorer for social media posts
- 📈 Real-time simulation of analysis pipeline
- 🎯 Detailed credibility scoring and explanations

## Getting Started

### Prerequisites

- Node.js 18+ installed
- npm or yarn package manager

### Installation

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
# or
yarn install
```

3. Run the development server:
```bash
npm run dev
# or
yarn dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Project Structure

```
frontend/
├── app/
│   ├── layout.tsx      # Root layout with metadata
│   ├── page.tsx        # Main dashboard component
│   └── globals.css     # Global styles and Tailwind imports
├── next.config.js      # Next.js configuration
├── tailwind.config.js  # Tailwind CSS configuration
├── tsconfig.json       # TypeScript configuration
└── package.json        # Dependencies and scripts
```

## Technologies

- **Next.js 14** - React framework with App Router
- **Tailwind CSS** - Utility-first CSS framework
- **Framer Motion** - Animation library
- **Lucide React** - Icon library
- **TypeScript** - Type safety

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Run ESLint

## Features Overview

### System Overview
- Project metadata and design goals
- Quick navigation to data explorer

### Input Data Explorer
- Browse sample social media posts
- Select posts for analysis
- View post metadata (likes, retweets, account info)

### Pipeline Stages
- 8-phase analysis pipeline visualization
- Real-time progress tracking
- Detailed phase information

### Analysis Report
- Credibility score visualization
- Agentic reasoning trace
- Detailed explanations and warnings

## Customization

The dashboard uses sample data from `SAMPLE_DATASET` in `app/page.tsx`. To connect to your backend API, modify the data fetching logic in the component.

## License

Part of the GenAI Social Media Credibility Analyzer research project.
