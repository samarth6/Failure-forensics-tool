import { Handle, Position, ReactFlow, type Edge, type Node, type NodeProps } from "@xyflow/react";
import type { Assessment, Diagnosis, Role, Span, StepName, Trace } from "../api";
import { ROLE_LABEL, STEPS, pct } from "../format";

type StepData = {
  step: StepName;
  role: Role;
  assessment?: Assessment;
  span?: Span;
  selected: boolean;
};
type StepNodeType = Node<StepData, "step">;

function StepNode({ data }: NodeProps<StepNodeType>) {
  const { step, role, assessment, span, selected } = data;
  const quality = role === "not_run" ? 0 : assessment?.local_quality ?? 1;
  return (
    <div className={`step-node role-${role}${selected ? " selected" : ""}`} title="Click to inspect this step">
      <Handle type="target" position={Position.Left} style={{ opacity: 0 }} />
      <div className="name">{step}</div>
      <div className="role">{ROLE_LABEL[role]}</div>
      <div className="bar">
        <div style={{ width: `${Math.round(quality * 100)}%` }} />
      </div>
      <div className="nums">
        <span>own checks {role === "not_run" ? "–" : pct(assessment?.local_quality ?? null)}</span>
        <span>flow {pct(assessment?.q_out ?? null)}</span>
      </div>
      <div className="nums">
        <span>{span ? `${(span.duration_ms / 1000).toFixed(1)}s${span.from_cache ? " · cached" : ""}` : "did not run"}</span>
        <span>{span?.self_confidence != null ? `self ${span.self_confidence}/5` : ""}</span>
      </div>
      <Handle type="source" position={Position.Right} style={{ opacity: 0 }} />
    </div>
  );
}

const nodeTypes = { step: StepNode };

function roleFor(step: StepName, trace: Trace, diagnosis: Diagnosis | null): Role {
  const assessment = diagnosis?.assessments.find((a) => a.step_name === step);
  if (assessment) return assessment.role;
  const span = trace.spans.find((s) => s.step_name === step);
  if (!span) return "not_run";
  return span.status === "error" ? "root_cause" : "healthy";
}

export default function TraceGraph({
  trace,
  diagnosis,
  selected,
  onSelect,
}: {
  trace: Trace;
  diagnosis: Diagnosis | null;
  selected: StepName;
  onSelect: (step: StepName) => void;
}) {
  const nodes: StepNodeType[] = STEPS.map((step, i) => ({
    id: step,
    type: "step",
    // Cards are ~216px wide; the rest of the step is the gap that holds the edge label.
    position: { x: i * 320, y: 0 },
    data: {
      step,
      role: roleFor(step, trace, diagnosis),
      assessment: diagnosis?.assessments.find((a) => a.step_name === step),
      span: trace.spans.find((s) => s.step_name === step),
      selected: step === selected,
    },
  }));

  // An edge carries quality from one step to the next. Edges into steps that inherited
  // damage are animated in the propagation colour: the path the error travelled.
  const edges: Edge[] = STEPS.slice(1).map((step, i) => {
    const target = nodes[i + 1].data;
    const carried = target.role === "propagation" || (target.role !== "not_run" && (target.assessment?.q_in ?? 1) < 0.6);
    return {
      id: `${STEPS[i]}-${step}`,
      source: STEPS[i],
      target: step,
      animated: carried,
      label: target.assessment ? `quality in ${pct(target.assessment.q_in)}` : undefined,
      style: {
        stroke: carried ? "var(--propagation)" : "var(--border)",
        strokeWidth: carried ? 2.5 : 1.5,
        strokeDasharray: target.role === "not_run" ? "4 4" : undefined,
      },
      labelStyle: { fontSize: 11, fill: "var(--muted)" },
      labelBgStyle: { fill: "var(--panel)" },
      labelBgPadding: [6, 3] as [number, number],
    };
  });

  return (
    <div className="graph-wrap">
      <ReactFlow
        key={trace.trace_id}
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodeClick={(_, node) => onSelect(node.id as StepName)}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
        zoomOnScroll={false}
        panOnScroll={false}
        panOnDrag={false}
        zoomOnPinch={false}
        zoomOnDoubleClick={false}
        preventScrolling={false}
        fitView
        fitViewOptions={{ padding: 0.15 }}
        proOptions={{ hideAttribution: true }}
      />
    </div>
  );
}

export function Legend() {
  const roles: Role[] = ["healthy", "low_confidence", "root_cause", "candidate", "propagation", "not_run"];
  return (
    <div className="legend">
      {roles.map((r) => (
        <span key={r} className={`role-${r}`}>
          <span className="dot" style={{ background: "currentColor" }} />
          <span style={{ color: "var(--muted)" }}>{ROLE_LABEL[r]}</span>
        </span>
      ))}
    </div>
  );
}
