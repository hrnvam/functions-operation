import { useState, useEffect, useRef, useCallback } from "react";
import Chart from "chart.js/auto";

function sigmoid(a, c, x) {
  return 1 / (1 + Math.exp(-a * (x - c)));
}

const MF_TYPES = {
  laplace: {
    label: "Лапласівська",
    defaultParams: { c: 0, b: 1 },
    paramDefs: [
      { id: "c", label: "c — центр", step: 0.1 },
      { id: "b", label: "b — масштаб (b > 0)", step: 0.1, min: 0.01 },
    ],
    compute: (p, x) => (p.b <= 0 ? 0 : Math.exp(-Math.abs(x - p.c) / p.b)),
    peak: (p) => p.c,
  },
  prod_sigmoid: {
    label: "Добуток сигмоїдних",
    defaultParams: { a1: -6, c1: 7, a2: 2, c2: 2 },
    paramDefs: [
      { id: "a1", label: "a₁ — крутизна 1", step: 0.1 },
      { id: "c1", label: "c₁ — поріг 1", step: 0.1 },
      { id: "a2", label: "a₂ — крутизна 2", step: 0.1 },
      { id: "c2", label: "c₂ — поріг 2", step: 0.1 },
    ],
    compute: (p, x) => sigmoid(p.a1, p.c1, x) * sigmoid(p.a2, p.c2, x),
    peak: (p) => (p.c1 + p.c2) / 2,
  },
  diff_sigmoid: {
    label: "Різниця сигмоїдних",
    defaultParams: { a1: 6, c1: 2, a2: 2, c2: 7 },
    paramDefs: [
      { id: "a1", label: "a₁ — крутизна 1", step: 0.1 },
      { id: "c1", label: "c₁ — поріг 1", step: 0.1 },
      { id: "a2", label: "a₂ — крутизна 2", step: 0.1 },
      { id: "c2", label: "c₂ — поріг 2", step: 0.1 },
    ],
    compute: (p, x) =>
      Math.max(0, sigmoid(p.a1, p.c1, x) - sigmoid(p.a2, p.c2, x)),
    peak: (p) => (p.c1 + p.c2) / 2,
  },
};

const OPS = [
  { id: "add", label: "Додавання (+)", fn: (x, y) => x + y },
  { id: "sub", label: "Віднімання (−)", fn: (x, y) => x - y },
  { id: "mul", label: "Множення (×)", fn: (x, y) => x * y },
  { id: "div", label: "Ділення (÷)", fn: (x, y) => (y !== 0 ? x / y : null) },
  { id: "max", label: "Максимум", fn: (x, y) => Math.max(x, y) },
  { id: "min", label: "Мінімум", fn: (x, y) => Math.min(x, y) },
];

function findBound(computeFn, start, direction, eps) {
  let step = 0.001;
  let x = start;
  let prev = x;
  for (let i = 0; i < 500000; i++) {
    x += direction * step;
    const mu = computeFn(x);
    if (mu < eps) {
      let lo = Math.min(prev, x),
        hi = Math.max(prev, x);
      for (let k = 0; k < 40; k++) {
        const mid = (lo + hi) / 2;
        computeFn(mid) >= eps ? (lo = mid) : (hi = mid);
      }
      return direction > 0 ? hi : lo;
    }
    prev = x;
    if (Math.abs(x - start) > 1 && i % 100 === 0)
      step = Math.min(step * 1.5, 0.5);
    if (Math.abs(x - start) > 500) return x;
  }
  return x;
}

function autoDiscretize(mfType, params, m, eps) {
  const mf = MF_TYPES[mfType];
  const computeFn = (x) => mf.compute(params, x);
  let peakX = mf.peak(params);

  if (computeFn(peakX) < eps) {
    let bestX = peakX;
    let maxMu = computeFn(peakX);
    for (let x = -100; x <= 100; x += 0.5) {
      const val = computeFn(x);
      if (val > maxMu) { maxMu = val; bestX = x; }
    }
    peakX = bestX;
  }

  const leftBound = findBound(computeFn, peakX, -1, eps);
  const rightBound = findBound(computeFn, peakX, +1, eps);

  const halfSteps = Math.max(2, Math.floor(m / 2));
  const leftStep = (peakX - leftBound) / halfSteps;
  const rightStep = (rightBound - peakX) / halfSteps;

  const xs = [];
  const mus = [];

  for (let i = 0; i <= halfSteps; i++) {
    const x = leftBound + i * leftStep;
    const mu = computeFn(x);
    if (mu >= eps) {
      xs.push(x);
      mus.push(mu);
    }
  }

  for (let i = 1; i <= halfSteps; i++) {
    const x = peakX + i * rightStep;
    const mu = computeFn(x);
    if (mu >= eps) {
      xs.push(x);
      mus.push(mu);
    }
  }

  return { xs, mus, xmin: leftBound, xmax: rightBound };
}

function maxMinConvolve(A, B, opFn) {
  const zMap = new Map();
  for (let i = 0; i < A.xs.length; i++) {
    for (let j = 0; j < B.xs.length; j++) {
      const z = opFn(A.xs[i], B.xs[j]);
      if (z === null || !isFinite(z)) continue;
      const zr = Math.round(z * 10000) / 10000;
      const mu = Math.min(A.mus[i], B.mus[j]);
      if (!zMap.has(zr) || zMap.get(zr) < mu) zMap.set(zr, mu);
    }
  }
  const zArr = Array.from(zMap.keys()).sort((a, b) => a - b);
  return { zs: zArr, mus: zArr.map((z) => zMap.get(z)) };
}

function denseCurve(mfType, params, xmin, xmax, pts = 300) {
  const mf = MF_TYPES[mfType];
  const out = [];
  for (let i = 0; i <= pts; i++) {
    const x = xmin + (i / pts) * (xmax - xmin);
    out.push({ x, y: Math.max(0, mf.compute(params, x)) });
  }
  return out;
}

function useChart(canvasRef, data) {
  const chartRef = useRef(null);
  useEffect(() => {
    if (!canvasRef.current || !data) return;
    if (chartRef.current) chartRef.current.destroy();
    chartRef.current = new Chart(canvasRef.current, {
      type: "scatter",
      data: {
        datasets: [
          {
            label: "A",
            data: data.curveA,
            borderColor: "#4f8ef7",
            backgroundColor: "rgba(79,142,247,0.12)",
            showLine: true,
            pointRadius: 0,
            borderWidth: 2.5,
            tension: 0.3,
            order: 4,
            fill: true,
          },
          {
            label: "B",
            data: data.curveB,
            borderColor: "#22c98e",
            backgroundColor: "rgba(34,201,142,0.12)",
            showLine: true,
            pointRadius: 0,
            borderWidth: 2.5,
            tension: 0.3,
            order: 3,
            fill: true,
          },
          {
            label: "C (Огинаюча)",
            data: data.envC,
            borderColor: "#ea580c",
            backgroundColor: "rgba(240,102,60,0.18)",
            showLine: true,
            pointRadius: 0,
            borderWidth: 3,
            tension: 0.15, // Зменшено tension, щоб не було петель
            order: 1,
            fill: true,
          },
          {
            label: "C (Сінглтони)",
            data: data.rawC,
            borderColor: "rgba(240,102,60,0.3)",
            backgroundColor: "transparent",
            showLine: true,
            pointRadius: 1.5,
            pointBackgroundColor: "rgba(240,102,60,0.5)",
            borderWidth: 1,
            tension: 0,
            order: 2,
            fill: false,
          },
          {
            label: "A точки",
            data: data.dotsA,
            borderColor: "#4f8ef7",
            backgroundColor: "#4f8ef7",
            showLine: false,
            pointRadius: 3.5,
            order: 0,
          },
          {
            label: "B точки",
            data: data.dotsB,
            borderColor: "#22c98e",
            backgroundColor: "#22c98e",
            showLine: false,
            pointRadius: 3.5,
            order: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 300 },
        plugins: {
          legend: { display: false },
          tooltip: {
            filter: (item) => item.datasetIndex < 4,
            callbacks: {
              label: (ctx) =>
                `${ctx.dataset.label}: (${ctx.parsed.x.toFixed(
                  4
                )}, ${ctx.parsed.y.toFixed(4)})`,
            },
          },
        },
        scales: {
          x: {
            title: {
              display: true,
              text: "x",
              color: "#6b7280",
              font: { size: 12 },
            },
            grid: { color: "rgba(0,0,0,0.06)" },
            ticks: { color: "#6b7280", font: { size: 11 } },
          },
          y: {
            min: 0,
            max: 1.08,
            title: {
              display: true,
              text: "μ(x)",
              color: "#6b7280",
              font: { size: 12 },
            },
            grid: { color: "rgba(0,0,0,0.06)" },
            ticks: { color: "#6b7280", font: { size: 11 }, stepSize: 0.2 },
          },
        },
      },
    });
    return () => {
      if (chartRef.current) chartRef.current.destroy();
    };
  }, [data]);
}

function NumInput({ label, value, onChange, step = 0.1, min }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
      <label
        style={{
          fontSize: 10,
          color: "#6b7280",
          letterSpacing: "0.06em",
          textTransform: "uppercase",
        }}
      >
        {label}
      </label>
      <input
        type="number"
        value={value}
        step={step}
        min={min}
        onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
        style={{
          background: "#ffffff",
          border: "1px solid #d1d5db",
          borderRadius: 5,
          color: "#111827",
          fontSize: 13,
          padding: "5px 7px",
          width: "100%",
          outline: "none",
          fontFamily: "monospace",
        }}
      />
    </div>
  );
}

function ParamsPanel({ title, color, params, setParams, mfType, rangeInfo }) {
  const mf = MF_TYPES[mfType];
  return (
    <div
      style={{
        background: "#ffffff",
        border: `1px solid ${color}40`,
        borderTop: `3px solid ${color}`,
        borderRadius: 10,
        padding: "13px 13px 11px",
        flex: 1,
        minWidth: 165,
        boxShadow: "0 1px 3px rgba(0,0,0,0.05)",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "baseline",
          gap: 8,
          marginBottom: 9,
        }}
      >
        <span
          style={{
            fontSize: 20,
            fontWeight: 700,
            color,
            fontFamily: "Georgia, serif",
          }}
        >
          {title}
        </span>
        {rangeInfo ? (
          <span
            style={{ fontSize: 10, color: "#6b7280", fontFamily: "monospace" }}
          >
            [{rangeInfo.xmin.toFixed(3)},&nbsp;{rangeInfo.xmax.toFixed(3)}
            ]&nbsp;·&nbsp;{rangeInfo.n}&nbsp;точок
          </span>
        ) : (
          <span style={{ fontSize: 10, color: "#9ca3af" }}>
            межі авто (за ε)
          </span>
        )}
      </div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: "7px 10px",
        }}
      >
        {mf.paramDefs.map((pd) => (
          <NumInput
            key={pd.id}
            label={pd.label}
            value={params[pd.id]}
            step={pd.step}
            min={pd.min}
            onChange={(v) => setParams((p) => ({ ...p, [pd.id]: v }))}
          />
        ))}
      </div>
    </div>
  );
}

export default function App() {
  const [mfType, setMfType] = useState("laplace");
  const [opId, setOpId] = useState("add");
  const [steps, setSteps] = useState(20);
  const [eps, setEps] = useState(0.01);

  const [paramsA, setParamsA] = useState({ ...MF_TYPES.laplace.defaultParams });
  const [paramsB, setParamsB] = useState({ ...MF_TYPES.laplace.defaultParams });

  const [error, setError] = useState("");
  const [chartData, setChartData] = useState(null);
  const [singletons, setSingletons] = useState(null);
  const [stats, setStats] = useState(null);
  const [rangeA, setRangeA] = useState(null);
  const [rangeB, setRangeB] = useState(null);

  const canvasRef = useRef(null);

  useEffect(() => {
    const def = MF_TYPES[mfType].defaultParams;
    setParamsA({ ...def });
    setParamsB({ ...def });
    setChartData(null);
    setSingletons(null);
    setStats(null);
    setRangeA(null);
    setRangeB(null);
  }, [mfType]);

  useChart(canvasRef, chartData);

  const compute = useCallback(() => {
    setError("");
    const op = OPS.find((o) => o.id === opId);
    const A = autoDiscretize(mfType, paramsA, steps, eps);
    const B = autoDiscretize(mfType, paramsB, steps, eps);

    if (A.xs.length === 0 || B.xs.length === 0) {
      setError("Жодна точка не перевищила ε. Зменшіть epsilon або перевірте параметри.");
      setChartData(null); 
      setSingletons(null);
      setStats(null); 
      setRangeA(null); 
      setRangeB(null);
      return;
    }
    const C = maxMinConvolve(A, B, op.fn);
    if (C.zs.length === 0) {
      setError("Результат порожній (можливо ділення на 0 або всі значення нескінченні).");
      setChartData(null); 
      setSingletons(null);
      setStats(null); 
      setRangeA(null); 
      setRangeB(null);
      return;
    }

    const padA = Math.max(0.1, (A.xmax - A.xmin) * 0.1);
    const padB = Math.max(0.1, (B.xmax - B.xmin) * 0.1);
    const cA = denseCurve(mfType, paramsA, A.xmin - padA, A.xmax + padA);
    const cB = denseCurve(mfType, paramsB, B.xmin - padB, B.xmax + padB);

    const rawC = C.zs.map((z, i) => ({ x: z, y: C.mus[i] }));

    const uniqueMus = Array.from(new Set(C.mus)).sort((a, b) => a - b);
    const leftBranch = [];
    const rightBranch = [];

    for (const muLevel of uniqueMus) {
      let minZ = Infinity;
      let maxZ = -Infinity;
      
      for (let i = 0; i < C.zs.length; i++) {
        if (C.mus[i] >= muLevel - 1e-9) {
          if (C.zs[i] < minZ) minZ = C.zs[i];
          if (C.zs[i] > maxZ) maxZ = C.zs[i];
        }
      }

      leftBranch.push({ x: minZ, y: muLevel });
      rightBranch.unshift({ x: maxZ, y: muLevel });
    }

    const envC = [...leftBranch, ...rightBranch];

    const dotsA = A.xs.map((x, i) => ({ x, y: A.mus[i] }));
    const dotsB = B.xs.map((x, i) => ({ x, y: B.mus[i] }));

    setChartData({ curveA: cA, curveB: cB, envC: envC, rawC, dotsA, dotsB });
    setSingletons({ zs: C.zs, mus: C.mus });
    setStats({ nA: A.xs.length, nB: B.xs.length, nC: C.zs.length });
    setRangeA({ xmin: A.xmin, xmax: A.xmax, n: A.xs.length });
    setRangeB({ xmin: B.xmin, xmax: B.xmax, n: B.xs.length });
  }, [mfType, paramsA, paramsB, steps, eps, opId]);

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#f4f6f8",
        color: "#1f2937",
        fontFamily: "'DM Sans','Segoe UI',sans-serif",
        padding: "18px 16px 48px",
      }}
    >
      <div
        style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 14 }}
      >
        {Object.entries(MF_TYPES).map(([k, v]) => (
          <button
            key={k}
            onClick={() => setMfType(k)}
            style={{
              padding: "6px 14px",
              borderRadius: 7,
              border: mfType === k ? "1px solid #4f8ef7" : "1px solid #d1d5db",
              background: mfType === k ? "#e0eaff" : "#ffffff",
              color: mfType === k ? "#2563eb" : "#4b5563",
              fontSize: 12,
              fontWeight: mfType === k ? 600 : 400,
              cursor: "pointer",
              transition: "all 0.12s",
              fontFamily: "inherit",
              boxShadow: "0 1px 2px rgba(0,0,0,0.05)",
            }}
          >
            {v.label}
          </button>
        ))}
      </div>

      <div
        style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 12 }}
      >
        <ParamsPanel
          title="A"
          color="#4f8ef7"
          mfType={mfType}
          params={paramsA}
          setParams={setParamsA}
          rangeInfo={rangeA}
        />
        <ParamsPanel
          title="B"
          color="#22c98e"
          mfType={mfType}
          params={paramsB}
          setParams={setParamsB}
          rangeInfo={rangeB}
        />

        <div
          style={{
            background: "#ffffff",
            border: "1px solid #e5e7eb",
            borderRadius: 10,
            padding: "13px",
            minWidth: 190,
            display: "flex",
            flexDirection: "column",
            gap: 10,
            boxShadow: "0 1px 3px rgba(0,0,0,0.05)",
          }}
        >
          <div>
            <div
              style={{
                fontSize: 10,
                color: "#6b7280",
                letterSpacing: "0.1em",
                textTransform: "uppercase",
                marginBottom: 6,
              }}
            >
              Операція
            </div>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: 4,
              }}
            >
              {OPS.map((op) => (
                <button
                  key={op.id}
                  onClick={() => setOpId(op.id)}
                  style={{
                    padding: "5px 5px",
                    borderRadius: 5,
                    border:
                      opId === op.id
                        ? "1px solid #f0663c"
                        : "1px solid #d1d5db",
                    background: opId === op.id ? "#fff5f2" : "#ffffff",
                    color: opId === op.id ? "#ea580c" : "#4b5563",
                    fontSize: 11,
                    cursor: "pointer",
                    textAlign: "center",
                    transition: "all 0.1s",
                    fontFamily: "inherit",
                  }}
                >
                  {op.label}
                </button>
              ))}
            </div>
          </div>

          <div
            style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}
          >
            <NumInput
              label="Кроки m"
              value={steps}
              step={1}
              min={4}
              onChange={(v) => setSteps(Math.max(4, Math.round(v)))}
            />
            <NumInput
              label="Epsilon ε"
              value={eps}
              step={0.005}
              min={0.001}
              onChange={setEps}
            />
          </div>

          <div style={{ fontSize: 10, color: "#6b7280", lineHeight: 1.6 }}>
            По&nbsp;
            <span
              style={{
                color: "#111827",
                fontFamily: "monospace",
                fontWeight: 500,
              }}
            >
              {Math.floor(steps / 2)}
            </span>
            &nbsp;кроків
          </div>

          <button
            onClick={compute}
            onMouseOver={(e) => (e.currentTarget.style.opacity = "0.85")}
            onMouseOut={(e) => (e.currentTarget.style.opacity = "1")}
            style={{
              padding: "9px",
              background: "#f0663c",
              border: "none",
              borderRadius: 7,
              color: "#fff",
              fontSize: 13,
              fontWeight: 700,
              cursor: "pointer",
              letterSpacing: "0.02em",
              fontFamily: "inherit",
              transition: "opacity 0.1s",
              boxShadow: "0 2px 4px rgba(240,102,60,0.3)",
            }}
          >
            Обрахувати
          </button>

          {error && (
            <div
              style={{
                fontSize: 11,
                color: "#dc2626",
                background: "#fef2f2",
                border: "1px solid #fca5a5",
                borderRadius: 5,
                padding: "6px 8px",
              }}
            >
              {error}
            </div>
          )}
        </div>
      </div>

      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 16,
          marginBottom: 7,
          flexWrap: "wrap",
        }}
      >
        {[
          ["#4f8ef7", "Число A"],
          ["#22c98e", "Число B"],
          ["#ea580c", "Огинаюча C (плавна)"],
          ["rgba(240,102,60,0.5)", "Сінглтони C"],
        ].map(([col, lbl]) => (
          <span
            key={lbl}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 6,
              color: "#4b5563",
              fontSize: 11,
              fontWeight: 500,
            }}
          >
            <span
              style={{
                width: 22,
                height: 4,
                background: col,
                borderRadius: 2,
                display: "inline-block",
              }}
            />
            {lbl}
          </span>
        ))}
        {stats && (
          <span
            style={{
              marginLeft: "auto",
              color: "#6b7280",
              fontFamily: "monospace",
              fontSize: 11,
              fontWeight: 500,
            }}
          >
            |A|={stats.nA} × |B|={stats.nB} → |C|={stats.nC} сінглтонів
          </span>
        )}
      </div>

      <div
        style={{
          background: "#ffffff",
          border: "1px solid #e5e7eb",
          borderRadius: 10,
          padding: "10px 10px 6px",
          marginBottom: 12,
          boxShadow: "0 2px 5px rgba(0,0,0,0.03)",
        }}
      >
        <div style={{ position: "relative", width: "100%", height: 300 }}>
          <canvas
            ref={canvasRef}
            role="img"
            aria-label="Графіки функцій належності нечітких чисел A, B та результату C"
          />
        </div>
      </div>

      {singletons && (
        <div
          style={{
            background: "#ffffff",
            border: "1px solid #e5e7eb",
            borderRadius: 10,
            padding: "11px 13px",
            boxShadow: "0 2px 5px rgba(0,0,0,0.03)",
          }}
        >
          <div
            style={{
              fontSize: 10,
              color: "#6b7280",
              letterSpacing: "0.12em",
              textTransform: "uppercase",
              marginBottom: 7,
              fontWeight: 600,
            }}
          >
            Результуюча нечітка множина C — {singletons.zs.length} сінглтонів
          </div>
          <div style={{ overflowX: "auto" }}>
            <table
              style={{
                borderCollapse: "collapse",
                fontSize: 11,
                fontFamily: "monospace",
              }}
            >
              <thead>
                <tr>
                  <td
                    style={{
                      padding: "4px 12px 4px 0",
                      color: "#111827",
                      borderBottom: "1px solid #e5e7eb",
                      whiteSpace: "nowrap",
                      fontWeight: 600,
                    }}
                  >
                    z
                  </td>
                  {singletons.zs.map((z, i) => (
                    <td
                      key={i}
                      style={{
                        padding: "4px 7px",
                        textAlign: "center",
                        borderBottom: "1px solid #e5e7eb",
                        color: "#4b5563",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {z.toFixed(3)}
                    </td>
                  ))}
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td
                    style={{
                      padding: "4px 12px 4px 0",
                      color: "#111827",
                      whiteSpace: "nowrap",
                      fontWeight: 600,
                    }}
                  >
                    μ(z)
                  </td>
                  {singletons.mus.map((mu, i) => (
                    <td
                      key={i}
                      style={{
                        padding: "4px 7px",
                        textAlign: "center",
                        whiteSpace: "nowrap",
                        color: "#9ca3af",
                        fontWeight: 400,
                      }}
                    >
                      {mu.toFixed(3)}
                    </td>
                  ))}
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
