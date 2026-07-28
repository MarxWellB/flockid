import React, { useState, useMemo } from "react";
import { MapPin, Radio, ListFilter, Settings, Egg, AlertTriangle, Activity, ScanLine, X, Camera } from "lucide-react";

const DATA = 
{"meta": {"n_birds": 15, "n_frames": 1800, "width": 960, "height": 540, "fleet_avg_risk": 58.9, "total_eggs": 19, "n_isolation_events": 170, "n_low_activity_events": 33}, "birds": [{"track_id": 16, "x": 714.6, "y": 124.0, "resolved_tag": "TAG-015", "fusion_confidence": 0.161, "egg_count": 5, "avg_egg_weight_g": 57.9, "avg_speed": 0.375, "repetitiveness": 0.075, "frames_tracked": 1618, "risk_score": 77.5, "behavior_score": 77.5, "top_evidence": [{"factor": "isolation", "contribution_pct": 30.0}, {"factor": "low_activity", "contribution_pct": 30.0}, {"factor": "low_speed", "contribution_pct": 17.5}]}, {"track_id": 2, "x": 477.4, "y": 499.8, "resolved_tag": "TAG-005", "fusion_confidence": 0.172, "egg_count": 3, "avg_egg_weight_g": 55.8, "avg_speed": 0.414, "repetitiveness": 0.047, "frames_tracked": 931, "risk_score": 77.2, "behavior_score": 77.2, "top_evidence": [{"factor": "isolation", "contribution_pct": 30.0}, {"factor": "low_activity", "contribution_pct": 30.0}, {"factor": "low_speed", "contribution_pct": 17.2}]}, {"track_id": 96, "x": 466.1, "y": 446.2, "resolved_tag": "TAG-004", "fusion_confidence": 0.269, "egg_count": 1, "avg_egg_weight_g": 57.1, "avg_speed": 0.419, "repetitiveness": 0.071, "frames_tracked": 635, "risk_score": 77.2, "behavior_score": 77.2, "top_evidence": [{"factor": "isolation", "contribution_pct": 30.0}, {"factor": "low_activity", "contribution_pct": 30.0}, {"factor": "low_speed", "contribution_pct": 17.2}]}, {"track_id": 156, "x": 768.0, "y": 81.0, "resolved_tag": "TAG-014", "fusion_confidence": 0.352, "egg_count": 1, "avg_egg_weight_g": 55.7, "avg_speed": 0.419, "repetitiveness": 0.163, "frames_tracked": 205, "risk_score": 77.2, "behavior_score": 77.2, "top_evidence": [{"factor": "isolation", "contribution_pct": 30.0}, {"factor": "low_activity", "contribution_pct": 30.0}, {"factor": "low_speed", "contribution_pct": 17.2}]}, {"track_id": 111, "x": 185.2, "y": 19.1, "resolved_tag": "TAG-012", "fusion_confidence": 0.284, "egg_count": 0, "avg_egg_weight_g": 0.0, "avg_speed": 0.432, "repetitiveness": 0.184, "frames_tracked": 629, "risk_score": 77.1, "behavior_score": 77.1, "top_evidence": [{"factor": "isolation", "contribution_pct": 30.0}, {"factor": "low_activity", "contribution_pct": 30.0}, {"factor": "low_speed", "contribution_pct": 17.1}]}, {"track_id": 41, "x": 205.0, "y": 66.3, "resolved_tag": "TAG-004", "fusion_confidence": 0.218, "egg_count": 2, "avg_egg_weight_g": 58.5, "avg_speed": 0.629, "repetitiveness": 0.406, "frames_tracked": 837, "risk_score": 75.8, "behavior_score": 75.8, "top_evidence": [{"factor": "isolation", "contribution_pct": 30.0}, {"factor": "low_activity", "contribution_pct": 30.0}, {"factor": "low_speed", "contribution_pct": 15.8}]}, {"track_id": 87, "x": 557.3, "y": 381.7, "resolved_tag": "TAG-013", "fusion_confidence": 1.0, "egg_count": 2, "avg_egg_weight_g": 60.6, "avg_speed": 1.316, "repetitiveness": 0.211, "frames_tracked": 108, "risk_score": 71.2, "behavior_score": 71.2, "top_evidence": [{"factor": "isolation", "contribution_pct": 30.0}, {"factor": "low_activity", "contribution_pct": 30.0}, {"factor": "low_speed", "contribution_pct": 11.2}]}, {"track_id": 17, "x": 494.4, "y": 450.8, "resolved_tag": "TAG-012", "fusion_confidence": 0.292, "egg_count": 5, "avg_egg_weight_g": 59.9, "avg_speed": 1.595, "repetitiveness": 0.479, "frames_tracked": 628, "risk_score": 69.4, "behavior_score": 69.4, "top_evidence": [{"factor": "isolation", "contribution_pct": 30.0}, {"factor": "low_activity", "contribution_pct": 30.0}, {"factor": "low_speed", "contribution_pct": 9.4}]}, {"track_id": 40, "x": 250.0, "y": 37.1, "resolved_tag": null, "fusion_confidence": 0, "egg_count": 0, "avg_egg_weight_g": 0, "avg_speed": 2.584, "repetitiveness": 0.926, "frames_tracked": 497, "risk_score": 52.8, "behavior_score": 52.8, "top_evidence": [{"factor": "isolation", "contribution_pct": 30.0}, {"factor": "repetitiveness", "contribution_pct": 20.0}, {"factor": "low_speed", "contribution_pct": 2.8}]}, {"track_id": 13, "x": 210.1, "y": 95.2, "resolved_tag": null, "fusion_confidence": 0, "egg_count": 0, "avg_egg_weight_g": 0, "avg_speed": 2.661, "repetitiveness": 0.925, "frames_tracked": 436, "risk_score": 52.3, "behavior_score": 52.3, "top_evidence": [{"factor": "isolation", "contribution_pct": 30.0}, {"factor": "repetitiveness", "contribution_pct": 20.0}, {"factor": "low_speed", "contribution_pct": 2.3}]}, {"track_id": 131, "x": 748.4, "y": 96.4, "resolved_tag": null, "fusion_confidence": 0, "egg_count": 0, "avg_egg_weight_g": 0, "avg_speed": 2.812, "repetitiveness": 0.937, "frames_tracked": 298, "risk_score": 51.3, "behavior_score": 51.3, "top_evidence": [{"factor": "isolation", "contribution_pct": 30.0}, {"factor": "repetitiveness", "contribution_pct": 20.0}, {"factor": "low_speed", "contribution_pct": 1.3}]}, {"track_id": 78, "x": 476.6, "y": 460.5, "resolved_tag": null, "fusion_confidence": 0, "egg_count": 0, "avg_egg_weight_g": 0, "avg_speed": 2.839, "repetitiveness": 0.955, "frames_tracked": 207, "risk_score": 51.1, "behavior_score": 51.1, "top_evidence": [{"factor": "isolation", "contribution_pct": 30.0}, {"factor": "repetitiveness", "contribution_pct": 20.0}, {"factor": "low_speed", "contribution_pct": 1.1}]}, {"track_id": 110, "x": 484.5, "y": 458.0, "resolved_tag": null, "fusion_confidence": 0, "egg_count": 0, "avg_egg_weight_g": 0, "avg_speed": 2.847, "repetitiveness": 0.975, "frames_tracked": 295, "risk_score": 51.0, "behavior_score": 51.0, "top_evidence": [{"factor": "isolation", "contribution_pct": 30.0}, {"factor": "repetitiveness", "contribution_pct": 20.0}, {"factor": "low_speed", "contribution_pct": 1.0}]}, {"track_id": 32, "x": 476.6, "y": 464.5, "resolved_tag": null, "fusion_confidence": 0, "egg_count": 0, "avg_egg_weight_g": 0, "avg_speed": 2.909, "repetitiveness": 0.907, "frames_tracked": 431, "risk_score": 50.6, "behavior_score": 50.6, "top_evidence": [{"factor": "isolation", "contribution_pct": 30.0}, {"factor": "repetitiveness", "contribution_pct": 20.0}, {"factor": "low_speed", "contribution_pct": 0.6}]}, {"track_id": 138, "x": 229.0, "y": 75.1, "resolved_tag": null, "fusion_confidence": 0, "egg_count": 0, "avg_egg_weight_g": 0, "avg_speed": 2.917, "repetitiveness": 0.957, "frames_tracked": 421, "risk_score": 50.6, "behavior_score": 50.6, "top_evidence": [{"factor": "isolation", "contribution_pct": 30.0}, {"factor": "repetitiveness", "contribution_pct": 20.0}, {"factor": "low_speed", "contribution_pct": 0.6}]}, {"track_id": 14, "x": 478.0, "y": 471.0, "resolved_tag": null, "fusion_confidence": 0, "egg_count": 0, "avg_egg_weight_g": 0, "avg_speed": 2.975, "repetitiveness": 0.941, "frames_tracked": 434, "risk_score": 50.2, "behavior_score": 50.2, "top_evidence": [{"factor": "isolation", "contribution_pct": 30.0}, {"factor": "repetitiveness", "contribution_pct": 20.0}, {"factor": "low_speed", "contribution_pct": 0.2}]}, {"track_id": 126, "x": 281.6, "y": 308.9, "resolved_tag": null, "fusion_confidence": 0, "egg_count": 0, "avg_egg_weight_g": 0, "avg_speed": 2.973, "repetitiveness": 0.923, "frames_tracked": 499, "risk_score": 50.2, "behavior_score": 50.2, "top_evidence": [{"factor": "isolation", "contribution_pct": 30.0}, {"factor": "repetitiveness", "contribution_pct": 20.0}, {"factor": "low_speed", "contribution_pct": 0.2}]}, {"track_id": 6, "x": 484.0, "y": 462.9, "resolved_tag": null, "fusion_confidence": 0, "egg_count": 0, "avg_egg_weight_g": 0, "avg_speed": 2.984, "repetitiveness": 0.965, "frames_tracked": 187, "risk_score": 50.1, "behavior_score": 50.1, "top_evidence": [{"factor": "isolation", "contribution_pct": 30.0}, {"factor": "repetitiveness", "contribution_pct": 20.0}, {"factor": "low_speed", "contribution_pct": 0.1}]}, {"track_id": 67, "x": 201.8, "y": 78.5, "resolved_tag": null, "fusion_confidence": 0, "egg_count": 0, "avg_egg_weight_g": 0, "avg_speed": 2.983, "repetitiveness": 0.859, "frames_tracked": 404, "risk_score": 50.1, "behavior_score": 50.1, "top_evidence": [{"factor": "isolation", "contribution_pct": 30.0}, {"factor": "repetitiveness", "contribution_pct": 20.0}, {"factor": "low_speed", "contribution_pct": 0.1}]}, {"track_id": 127, "x": 469.6, "y": 452.0, "resolved_tag": null, "fusion_confidence": 0, "egg_count": 0, "avg_egg_weight_g": 0, "avg_speed": 2.991, "repetitiveness": 0.965, "frames_tracked": 317, "risk_score": 50.1, "behavior_score": 50.1, "top_evidence": [{"factor": "isolation", "contribution_pct": 30.0}, {"factor": "repetitiveness", "contribution_pct": 20.0}, {"factor": "low_speed", "contribution_pct": 0.1}]}, {"track_id": 166, "x": 582.1, "y": 168.9, "resolved_tag": null, "fusion_confidence": 0, "egg_count": 0, "avg_egg_weight_g": 0, "avg_speed": 2.979, "repetitiveness": 0.901, "frames_tracked": 112, "risk_score": 50.1, "behavior_score": 50.1, "top_evidence": [{"factor": "isolation", "contribution_pct": 30.0}, {"factor": "repetitiveness", "contribution_pct": 20.0}, {"factor": "low_speed", "contribution_pct": 0.1}]}, {"track_id": 3, "x": 767.9, "y": 79.0, "resolved_tag": null, "fusion_confidence": 0, "egg_count": 0, "avg_egg_weight_g": 0, "avg_speed": 3.013, "repetitiveness": 0.92, "frames_tracked": 376, "risk_score": 50.0, "behavior_score": 50.0, "top_evidence": [{"factor": "isolation", "contribution_pct": 30.0}, {"factor": "repetitiveness", "contribution_pct": 20.0}, {"factor": "low_activity", "contribution_pct": 0.0}]}, {"track_id": 18, "x": 179.0, "y": 78.5, "resolved_tag": null, "fusion_confidence": 0, "egg_count": 0, "avg_egg_weight_g": 0, "avg_speed": 3.02, "repetitiveness": 0.894, "frames_tracked": 188, "risk_score": 50.0, "behavior_score": 50.0, "top_evidence": [{"factor": "isolation", "contribution_pct": 30.0}, {"factor": "repetitiveness", "contribution_pct": 20.0}, {"factor": "low_activity", "contribution_pct": 0.0}]}, {"track_id": 34, "x": 765.1, "y": 77.0, "resolved_tag": null, "fusion_confidence": 0, "egg_count": 0, "avg_egg_weight_g": 0, "avg_speed": 2.995, "repetitiveness": 0.914, "frames_tracked": 248, "risk_score": 50.0, "behavior_score": 50.0, "top_evidence": [{"factor": "isolation", "contribution_pct": 30.0}, {"factor": "repetitiveness", "contribution_pct": 20.0}, {"factor": "low_activity", "contribution_pct": 0.0}]}], "nests": [{"nest_id": "nido_A", "x": 192.0, "y": 81.0, "radius": 35.0}, {"nest_id": "nido_B", "x": 768.0, "y": 81.0, "radius": 35.0}, {"nest_id": "nido_C", "x": 480.0, "y": 459.0, "radius": 35.0}], "events": [{"event_type": "isolation", "entity_id": 152, "confidence": 0.6, "evidence": {"nearest_neighbor_distance_px": 96.7, "sustained_frames": 15}, "frame_idx": 1798}, {"event_type": "isolation", "entity_id": 168, "confidence": 0.69, "evidence": {"nearest_neighbor_distance_px": 110.7, "sustained_frames": 15}, "frame_idx": 1788}, {"event_type": "isolation", "entity_id": 166, "confidence": 0.7, "evidence": {"nearest_neighbor_distance_px": 111.7, "sustained_frames": 15}, "frame_idx": 1782}, {"event_type": "isolation", "entity_id": 170, "confidence": 0.69, "evidence": {"nearest_neighbor_distance_px": 111.0, "sustained_frames": 15}, "frame_idx": 1772}, {"event_type": "isolation", "entity_id": 156, "confidence": 0.64, "evidence": {"nearest_neighbor_distance_px": 102.1, "sustained_frames": 15}, "frame_idx": 1763}, {"event_type": "low_activity", "entity_id": 156, "confidence": 0.7, "evidence": {"avg_speed_px_per_frame": 0.197, "window_frames": 30}, "frame_idx": 1757}, {"event_type": "isolation", "entity_id": 138, "confidence": 0.83, "evidence": {"nearest_neighbor_distance_px": 132.4, "sustained_frames": 15}, "frame_idx": 1729}, {"event_type": "isolation", "entity_id": 158, "confidence": 0.83, "evidence": {"nearest_neighbor_distance_px": 132.4, "sustained_frames": 15}, "frame_idx": 1729}, {"event_type": "isolation", "entity_id": 156, "confidence": 0.73, "evidence": {"nearest_neighbor_distance_px": 117.1, "sustained_frames": 15}, "frame_idx": 1725}, {"event_type": "isolation", "entity_id": 166, "confidence": 0.65, "evidence": {"nearest_neighbor_distance_px": 104.2, "sustained_frames": 15}, "frame_idx": 1724}, {"event_type": "low_activity", "entity_id": 111, "confidence": 0.7, "evidence": {"avg_speed_px_per_frame": 0.0, "window_frames": 30}, "frame_idx": 1723}, {"event_type": "isolation", "entity_id": 152, "confidence": 0.94, "evidence": {"nearest_neighbor_distance_px": 151.1, "sustained_frames": 15}, "frame_idx": 1714}, {"event_type": "low_activity", "entity_id": 159, "confidence": 0.7, "evidence": {"avg_speed_px_per_frame": 0.0, "window_frames": 30}, "frame_idx": 1713}, {"event_type": "isolation", "entity_id": 111, "confidence": 1.0, "evidence": {"nearest_neighbor_distance_px": 192.2, "sustained_frames": 15}, "frame_idx": 1711}, {"event_type": "isolation", "entity_id": 126, "confidence": 0.74, "evidence": {"nearest_neighbor_distance_px": 117.9, "sustained_frames": 15}, "frame_idx": 1688}, {"event_type": "isolation", "entity_id": 138, "confidence": 0.94, "evidence": {"nearest_neighbor_distance_px": 149.6, "sustained_frames": 15}, "frame_idx": 1682}, {"event_type": "isolation", "entity_id": 163, "confidence": 0.52, "evidence": {"nearest_neighbor_distance_px": 83.8, "sustained_frames": 15}, "frame_idx": 1680}, {"event_type": "isolation", "entity_id": 158, "confidence": 0.59, "evidence": {"nearest_neighbor_distance_px": 95.0, "sustained_frames": 15}, "frame_idx": 1676}, {"event_type": "isolation", "entity_id": 155, "confidence": 0.83, "evidence": {"nearest_neighbor_distance_px": 132.9, "sustained_frames": 15}, "frame_idx": 1669}, {"event_type": "low_activity", "entity_id": 111, "confidence": 0.7, "evidence": {"avg_speed_px_per_frame": 0.14, "window_frames": 30}, "frame_idx": 1659}, {"event_type": "isolation", "entity_id": 138, "confidence": 0.64, "evidence": {"nearest_neighbor_distance_px": 102.8, "sustained_frames": 15}, "frame_idx": 1644}, {"event_type": "isolation", "entity_id": 145, "confidence": 0.6, "evidence": {"nearest_neighbor_distance_px": 96.7, "sustained_frames": 15}, "frame_idx": 1632}, {"event_type": "isolation", "entity_id": 152, "confidence": 0.94, "evidence": {"nearest_neighbor_distance_px": 150.7, "sustained_frames": 15}, "frame_idx": 1632}, {"event_type": "isolation", "entity_id": 153, "confidence": 1.0, "evidence": {"nearest_neighbor_distance_px": 160.9, "sustained_frames": 15}, "frame_idx": 1626}, {"event_type": "isolation", "entity_id": 154, "confidence": 0.59, "evidence": {"nearest_neighbor_distance_px": 94.9, "sustained_frames": 15}, "frame_idx": 1594}, {"event_type": "low_activity", "entity_id": 111, "confidence": 0.7, "evidence": {"avg_speed_px_per_frame": 0.0, "window_frames": 30}, "frame_idx": 1581}, {"event_type": "isolation", "entity_id": 152, "confidence": 0.54, "evidence": {"nearest_neighbor_distance_px": 87.0, "sustained_frames": 15}, "frame_idx": 1569}, {"event_type": "isolation", "entity_id": 96, "confidence": 0.75, "evidence": {"nearest_neighbor_distance_px": 120.5, "sustained_frames": 15}, "frame_idx": 1560}, {"event_type": "low_activity", "entity_id": 16, "confidence": 0.7, "evidence": {"avg_speed_px_per_frame": 0.0, "window_frames": 30}, "frame_idx": 1559}, {"event_type": "isolation", "entity_id": 151, "confidence": 0.7, "evidence": {"nearest_neighbor_distance_px": 111.3, "sustained_frames": 15}, "frame_idx": 1557}]}
;

const CAMERAS = [
  { id: "cam_NO", x: 40, y: 40 },
  { id: "cam_NE", x: DATA.meta.width - 40, y: 40 },
  { id: "cam_SO", x: 40, y: DATA.meta.height - 40 },
  { id: "cam_SE", x: DATA.meta.width - 40, y: DATA.meta.height - 40 },
];

function riskColor(score) {
  if (score >= 70) return "#E85D4E";
  if (score >= 55) return "#E8A33D";
  return "#6FE7D9";
}

function riskLabel(score) {
  if (score >= 70) return "RETIRAR / OBSERVAR";
  if (score >= 55) return "VIGILAR";
  return "NORMAL";
}

function EvidenceBar({ label, pct }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, fontFamily: "'IBM Plex Mono', monospace", fontSize: 11 }}>
      <span style={{ width: 92, color: "#8A9891", flexShrink: 0, textTransform: "uppercase", letterSpacing: 0.5 }}>{label}</span>
      <div style={{ flex: 1, height: 5, background: "#1C2925", borderRadius: 2, overflow: "hidden" }}>
        <div style={{ width: `${Math.min(pct * 2.2, 100)}%`, height: "100%", background: "#E8A33D" }} />
      </div>
      <span style={{ width: 34, textAlign: "right", color: "#E9EDE9" }}>{pct.toFixed(0)}%</span>
    </div>
  );
}

function BirdHUDCard({ bird, onClose }) {
  if (!bird) return null;
  const color = riskColor(bird.risk_score);
  return (
    <div
      style={{
        position: "absolute",
        left: `${(bird.x / DATA.meta.width) * 100}%`,
        top: `${(bird.y / DATA.meta.height) * 100}%`,
        transform: "translate(18px, -50%)",
        zIndex: 30,
        width: 250,
        pointerEvents: "auto",
      }}
    >
      <div
        style={{
          background: "rgba(13, 21, 18, 0.94)",
          border: `1px solid ${color}`,
          borderRadius: 4,
          padding: "12px 14px",
          boxShadow: `0 0 0 1px rgba(0,0,0,0.4), 0 8px 24px rgba(0,0,0,0.5), 0 0 18px ${color}22`,
          backdropFilter: "blur(2px)",
        }}
      >
        <div style={{ position: "absolute", top: -1, left: -1, width: 10, height: 10, borderTop: `2px solid ${color}`, borderLeft: `2px solid ${color}` }} />
        <div style={{ position: "absolute", bottom: -1, right: -1, width: 10, height: 10, borderBottom: `2px solid ${color}`, borderRight: `2px solid ${color}` }} />

        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
          <div>
            <div style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: 15, fontWeight: 600, color: "#E9EDE9", letterSpacing: 0.3 }}>
              {bird.resolved_tag || `TRACK-${bird.track_id}`}
            </div>
            <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 10, color: "#8A9891", marginTop: 1 }}>
              visual_id #{bird.track_id}
            </div>
          </div>
          <button onClick={onClose} style={{ background: "none", border: "none", color: "#8A9891", cursor: "pointer", padding: 2 }}>
            <X size={14} />
          </button>
        </div>

        <div style={{ display: "flex", alignItems: "baseline", gap: 6, marginTop: 10 }}>
          <span style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: 28, fontWeight: 700, color }}>
            {bird.risk_score.toFixed(0)}
          </span>
          <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 10, color, letterSpacing: 0.5 }}>
            {riskLabel(bird.risk_score)}
          </span>
        </div>

        <div style={{ display: "flex", gap: 14, marginTop: 10, fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: "#C9D2CD" }}>
          <div>
            <div style={{ color: "#8A9891", fontSize: 9, textTransform: "uppercase" }}>Huevos</div>
            <div style={{ fontSize: 14, color: "#E9EDE9" }}>{bird.egg_count}</div>
          </div>
          <div>
            <div style={{ color: "#8A9891", fontSize: 9, textTransform: "uppercase" }}>Peso prom.</div>
            <div style={{ fontSize: 14, color: "#E9EDE9" }}>{bird.avg_egg_weight_g ? `${bird.avg_egg_weight_g}g` : "—"}</div>
          </div>
          <div>
            <div style={{ color: "#8A9891", fontSize: 9, textTransform: "uppercase" }}>Confianza ID</div>
            <div style={{ fontSize: 14, color: "#E9EDE9" }}>{(bird.fusion_confidence * 100).toFixed(0)}%</div>
          </div>
        </div>

        <div style={{ marginTop: 12, display: "flex", flexDirection: "column", gap: 6 }}>
          {bird.top_evidence.map((e, i) => (
            <EvidenceBar key={i} label={e.factor.replace("_", " ")} pct={e.contribution_pct} />
          ))}
        </div>
      </div>
    </div>
  );
}

export default function FlockIDDashboard() {
  const [selectedId, setSelectedId] = useState(DATA.birds[0]?.track_id ?? null);
  const [hoveredId, setHoveredId] = useState(null);
  const [tab, setTab] = useState("map");

  const activeId = hoveredId ?? selectedId;
  const activeBird = useMemo(() => DATA.birds.find((b) => b.track_id === activeId), [activeId]);
  const watchlist = useMemo(() => [...DATA.birds].sort((a, b) => b.risk_score - a.risk_score).slice(0, 10), []);

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#0D1512",
        color: "#E9EDE9",
        fontFamily: "'Inter', sans-serif",
        display: "flex",
      }}
    >
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
        * { box-sizing: border-box; }
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-thumb { background: #263630; border-radius: 3px; }
        .navbtn { transition: background 0.15s, color 0.15s; }
        .navbtn:hover { background: #1C2925; }
        .birddot { transition: r 0.15s, filter 0.15s; cursor: pointer; }
        .watchcard { transition: border-color 0.15s, transform 0.15s; cursor: pointer; }
        .watchcard:hover { transform: translateY(-2px); }
      `}</style>

      {/* Nav rail */}
      <div style={{ width: 60, borderRight: "1px solid #1C2925", display: "flex", flexDirection: "column", alignItems: "center", padding: "18px 0", gap: 4, flexShrink: 0 }}>
        <div style={{ width: 30, height: 30, border: "1.5px solid #E8A33D", borderRadius: 6, display: "flex", alignItems: "center", justifyContent: "center", marginBottom: 20 }}>
          <ScanLine size={16} color="#E8A33D" />
        </div>
        {[
          { icon: MapPin, key: "map", label: "Mapa" },
          { icon: ListFilter, key: "watchlist", label: "Lista" },
          { icon: Camera, key: "cams", label: "Cámaras" },
          { icon: Settings, key: "settings", label: "Config" },
        ].map(({ icon: Icon, key }) => (
          <button
            key={key}
            onClick={() => setTab(key)}
            className="navbtn"
            style={{
              width: 40, height: 40, borderRadius: 8, border: "none",
              background: tab === key ? "#1C2925" : "transparent",
              color: tab === key ? "#E8A33D" : "#8A9891",
              display: "flex", alignItems: "center", justifyContent: "center", cursor: "pointer",
            }}
          >
            <Icon size={17} />
          </button>
        ))}
      </div>

      {/* Main content */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>
        {/* Top bar */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "16px 24px", borderBottom: "1px solid #1C2925" }}>
          <div>
            <div style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: 19, fontWeight: 700, letterSpacing: 0.5 }}>
              FLOCK<span style={{ color: "#E8A33D" }}>ID</span>
            </div>
            <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: "#8A9891", marginTop: 1 }}>
              Galpón 3 · Lote #114 · {DATA.meta.n_frames.toLocaleString()} frames analizados
            </div>
          </div>
          <div style={{ display: "flex", gap: 28 }}>
            <KPI label="Riesgo de parvada" value={DATA.meta.fleet_avg_risk.toFixed(0)} color={riskColor(DATA.meta.fleet_avg_risk)} icon={Activity} />
            <KPI label="Huevos (sesión)" value={DATA.meta.total_eggs} color="#6FE7D9" icon={Egg} />
            <KPI label="Eventos aislamiento" value={DATA.meta.n_isolation_events} color="#E8A33D" icon={AlertTriangle} />
            <KPI label="Baja actividad" value={DATA.meta.n_low_activity_events} color="#E8A33D" icon={AlertTriangle} />
          </div>
        </div>

        {/* Body */}
        <div style={{ flex: 1, display: "flex", minHeight: 0 }}>
          {/* Map panel */}
          <div style={{ flex: 1, position: "relative", padding: 20, minWidth: 0 }}>
            <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 10.5, color: "#8A9891", marginBottom: 10, letterSpacing: 0.5, display: "flex", justifyContent: "space-between" }}>
              <span>VISTA DE GALPÓN — CONSENSO 4 CÁMARAS</span>
              <span>{DATA.birds.length} aves con perfil activo</span>
            </div>
            <div
              style={{
                position: "relative",
                width: "100%",
                aspectRatio: `${DATA.meta.width} / ${DATA.meta.height}`,
                background: "radial-gradient(ellipse at center, #14201B 0%, #0F1815 100%)",
                border: "1px solid #1C2925",
                borderRadius: 6,
                overflow: "hidden",
              }}
              onClick={() => setSelectedId(null)}
            >
              {/* grid */}
              <svg width="100%" height="100%" style={{ position: "absolute", inset: 0 }}>
                <defs>
                  <pattern id="grid" width="5%" height="8.9%" patternUnits="userSpaceOnUse">
                    <path d="M 0 0 L 0 0" />
                  </pattern>
                </defs>
                {Array.from({ length: 19 }).map((_, i) => (
                  <line key={`v${i}`} x1={`${(i + 1) * 5}%`} y1="0" x2={`${(i + 1) * 5}%`} y2="100%" stroke="#182420" strokeWidth="1" />
                ))}
                {Array.from({ length: 10 }).map((_, i) => (
                  <line key={`h${i}`} x1="0" y1={`${(i + 1) * 9.1}%`} x2="100%" y2={`${(i + 1) * 9.1}%`} stroke="#182420" strokeWidth="1" />
                ))}

                {/* camera coverage */}
                {CAMERAS.map((c) => (
                  <circle
                    key={c.id}
                    cx={`${(c.x / DATA.meta.width) * 100}%`}
                    cy={`${(c.y / DATA.meta.height) * 100}%`}
                    r="30%"
                    fill="none"
                    stroke="#6FE7D9"
                    strokeOpacity="0.14"
                    strokeWidth="1"
                    strokeDasharray="3 4"
                  />
                ))}
                {CAMERAS.map((c) => (
                  <g key={c.id + "dot"}>
                    <circle cx={`${(c.x / DATA.meta.width) * 100}%`} cy={`${(c.y / DATA.meta.height) * 100}%`} r="4" fill="#6FE7D9" />
                  </g>
                ))}

                {/* nests */}
                {DATA.nests.map((n) => (
                  <rect
                    key={n.nest_id}
                    x={`${((n.x - n.radius) / DATA.meta.width) * 100}%`}
                    y={`${((n.y - n.radius) / DATA.meta.height) * 100}%`}
                    width={`${(n.radius * 2 / DATA.meta.width) * 100}%`}
                    height={`${(n.radius * 2 / DATA.meta.height) * 100}%`}
                    fill="none"
                    stroke="#E8A33D"
                    strokeOpacity="0.4"
                    strokeWidth="1"
                    rx="4"
                  />
                ))}

                {/* birds */}
                {DATA.birds.map((b) => (
                  <circle
                    key={b.track_id}
                    className="birddot"
                    cx={`${(b.x / DATA.meta.width) * 100}%`}
                    cy={`${(b.y / DATA.meta.height) * 100}%`}
                    r={activeId === b.track_id ? 7 : 5}
                    fill={riskColor(b.risk_score)}
                    stroke="#0D1512"
                    strokeWidth="1.5"
                    style={{ filter: activeId === b.track_id ? `drop-shadow(0 0 6px ${riskColor(b.risk_score)})` : "none" }}
                    onMouseEnter={() => setHoveredId(b.track_id)}
                    onMouseLeave={() => setHoveredId(null)}
                    onClick={(e) => { e.stopPropagation(); setSelectedId(b.track_id); }}
                  />
                ))}
              </svg>

              <BirdHUDCard bird={activeBird} onClose={() => setSelectedId(null)} />
            </div>

            <div style={{ display: "flex", gap: 18, marginTop: 12, fontFamily: "'IBM Plex Mono', monospace", fontSize: 10.5, color: "#8A9891" }}>
              <Legend color="#6FE7D9" label="Normal (<55)" />
              <Legend color="#E8A33D" label="Vigilar (55-69)" />
              <Legend color="#E85D4E" label="Retirar/observar (≥70)" />
              <Legend color="#E8A33D" label="Zona de nido" outline />
              <Legend color="#6FE7D9" label="Cobertura de cámara" outline />
            </div>
          </div>

          {/* Right panel: event feed */}
          <div style={{ width: 300, borderLeft: "1px solid #1C2925", padding: "16px 18px", overflowY: "auto", flexShrink: 0 }}>
            <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 10.5, color: "#8A9891", marginBottom: 12, letterSpacing: 0.5 }}>
              EVENTOS RECIENTES (BEHAVIOR ENGINE)
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {DATA.events.slice(0, 20).map((e, i) => {
                const bird = DATA.birds.find((b) => b.track_id === e.entity_id);
                return (
                  <div
                    key={i}
                    onClick={() => bird && setSelectedId(bird.track_id)}
                    style={{
                      padding: "8px 10px", background: "#141F1A", border: "1px solid #1C2925", borderRadius: 4,
                      cursor: bird ? "pointer" : "default",
                    }}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 10.5, color: e.event_type === "isolation" ? "#E8A33D" : "#E85D4E", textTransform: "uppercase", letterSpacing: 0.5 }}>
                        {e.event_type === "isolation" ? "Aislamiento" : "Baja actividad"}
                      </span>
                      <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 9.5, color: "#5C6B64" }}>#{e.frame_idx}</span>
                    </div>
                    <div style={{ fontSize: 11.5, color: "#C9D2CD", marginTop: 3 }}>
                      {bird ? (bird.resolved_tag || `track_id ${bird.track_id}`) : `track_id ${e.entity_id}`}
                      {e.evidence.nearest_neighbor_distance_px && ` — vecino a ${e.evidence.nearest_neighbor_distance_px}px`}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Watchlist strip */}
        <div style={{ borderTop: "1px solid #1C2925", padding: "14px 24px" }}>
          <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 10.5, color: "#8A9891", marginBottom: 10, letterSpacing: 0.5 }}>
            LISTA DE ATENCIÓN — ORDENADO POR RIESGO
          </div>
          <div style={{ display: "flex", gap: 10, overflowX: "auto", paddingBottom: 4 }}>
            {watchlist.map((b) => (
              <div
                key={b.track_id}
                className="watchcard"
                onClick={() => setSelectedId(b.track_id)}
                style={{
                  minWidth: 148, flexShrink: 0, padding: "10px 12px", borderRadius: 5,
                  background: "#141F1A",
                  border: `1px solid ${selectedId === b.track_id ? riskColor(b.risk_score) : "#1C2925"}`,
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                  <span style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: 13, fontWeight: 600 }}>
                    {b.resolved_tag || `#${b.track_id}`}
                  </span>
                  <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 15, fontWeight: 600, color: riskColor(b.risk_score) }}>
                    {b.risk_score.toFixed(0)}
                  </span>
                </div>
                <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 9.5, color: "#8A9891", marginTop: 3 }}>
                  {riskLabel(b.risk_score)}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function KPI({ label, value, color, icon: Icon }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <Icon size={15} color={color} />
      <div>
        <div style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: 17, fontWeight: 700, lineHeight: 1, color }}>{value}</div>
        <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 9, color: "#8A9891", marginTop: 2, whiteSpace: "nowrap" }}>{label}</div>
      </div>
    </div>
  );
}

function Legend({ color, label, outline }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
      <span style={{
        width: 8, height: 8, borderRadius: outline ? 2 : 8,
        background: outline ? "none" : color,
        border: outline ? `1px solid ${color}` : "none",
      }} />
      {label}
    </div>
  );
}
