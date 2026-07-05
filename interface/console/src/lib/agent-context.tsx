"use client";

import React, { createContext, useContext, useState, useEffect } from "react";
import { client } from "./api";

interface AgentContextType {
  currentAgent: string | null;
  setCurrentAgent: (agent: string | null) => void;
  agents: string[];
  loadAgents: () => Promise<void>;
}

const AgentContext = createContext<AgentContextType | undefined>(undefined);

export function AgentProvider({ children }: { children: React.ReactNode }) {
  const [currentAgent, setCurrentAgent] = useState<string | null>(null);
  const [agents, setAgents] = useState<string[]>([]);

  const loadAgents = async () => {
    try {
      const data = await client.listBanks();
      // Extract bank_id from each bank object
      const agentIds = data.banks?.map((agent: any) => agent.bank_id) || [];
      setAgents(agentIds);
    } catch (error) {
      console.error("Error loading agents:", error);
    }
  };

  useEffect(() => {
    loadAgents();
  }, []);

  return (
    <AgentContext.Provider value={{ currentAgent, setCurrentAgent, agents, loadAgents }}>
      {children}
    </AgentContext.Provider>
  );
}

export function useAgent() {
  const context = useContext(AgentContext);
  if (context === undefined) {
    throw new Error("useAgent must be used within an AgentProvider");
  }
  return context;
}
