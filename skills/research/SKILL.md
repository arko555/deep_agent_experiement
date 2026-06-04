---
name: research
description: >
  Conducts comprehensive web research on any topic using search APIs 
  and web scraping. Use when the user asks for research, information 
  gathering, competitive analysis, or needs current data from the web.
license: MIT
compatibility: Requires internet access and Tavily API key
allowed-tools: internet_search Read Write
---

# Web Research Skill

## Overview
This skill provides structured web research capabilities. It handles breaking down complex research questions into targeted search queries and synthesizing findings.

## Instructions
1. Parse the research query to identify key topics and missing information.
2. Use the `internet_search` tool to gather initial results.
3. If results are insufficient, try alternative search queries.
4. Synthesize the findings into a clear, structured summary.
5. Save the detailed findings to the `./workspace` directory as a markdown file.

## Examples
### Input
"Research the latest developments in quantum computing"

### Expected Behavior
The agent should conduct 3-5 targeted searches, summarize the key breakthroughs (e.g., error correction, new qubit types), and write a report to `workspace/quantum_research.md`.
