//! Synthetic dataset generator for one-compartment oral PK model.
//!
//! This module generates test data with known population parameters for validation.
//! Uses simple ODE solver (RK4) to simulate individual PK profiles.

use std::f64::consts::PI;

/// True population parameters used to generate synthetic data
pub struct TrueParams {
    pub cl_pop: f64,     // Typical clearance [L/h]
    pub v_pop: f64,      // Typical volume [L]
    pub ka_pop: f64,     // Typical absorption rate [1/h]
    pub omega_cl: f64,   // SD of log(CL) random effects
    pub omega_v: f64,    // SD of log(V) random effects
    pub omega_ka: f64,   // SD of log(Ka) random effects
    pub sigma_prop: f64, // Proportional residual error SD
}

impl Default for TrueParams {
    fn default() -> Self {
        Self {
            cl_pop: 10.0,
            v_pop: 50.0,
            ka_pop: 1.0,
            omega_cl: 0.3,
            omega_v: 0.2,
            omega_ka: 0.4,
            sigma_prop: 0.15,
        }
    }
}

/// CSV row for NONMEM-style dataset
#[derive(Debug, Clone)]
pub struct DataRow {
    pub id: usize,        // Subject ID
    pub time: f64,        // Time [h]
    pub dv: Option<f64>,  // Observed concentration [mg/L] (None for dose rows)
    pub wt: f64,          // Body weight [kg]
    pub evid: u8,         // Event ID: 0=observation, 1=dose
    pub amt: Option<f64>, // Dose amount [mg] (None for observation rows)
}

/// ODE state for one-compartment oral PK
#[derive(Debug, Clone, Copy)]
struct PKState {
    a_gut: f64,     // Amount in gut [mg]
    a_central: f64, // Amount in central [mg]
}

impl PKState {
    fn derivative(&self, ka: f64, cl: f64, v: f64) -> Self {
        PKState {
            a_gut: -ka * self.a_gut,
            a_central: ka * self.a_gut - (cl / v) * self.a_central,
        }
    }

    fn add(&self, rhs: &PKState, scale: f64) -> Self {
        PKState {
            a_gut: self.a_gut + scale * rhs.a_gut,
            a_central: self.a_central + scale * rhs.a_central,
        }
    }
}

/// Simple RK4 ODE solver
fn rk4_step(state: PKState, dt: f64, ka: f64, cl: f64, v: f64) -> PKState {
    let k1 = state.derivative(ka, cl, v);
    let k2 = state.add(&k1, dt * 0.5).derivative(ka, cl, v);
    let k3 = state.add(&k2, dt * 0.5).derivative(ka, cl, v);
    let k4 = state.add(&k3, dt).derivative(ka, cl, v);

    state
        .add(&k1, dt / 6.0)
        .add(&k2, dt / 3.0)
        .add(&k3, dt / 3.0)
        .add(&k4, dt / 6.0)
}

/// Solve PK ODEs from time 0 to target time
fn solve_pk(ka: f64, cl: f64, v: f64, dose: f64, target_time: f64) -> f64 {
    let mut state = PKState {
        a_gut: dose,
        a_central: 0.0,
    };

    let dt: f64 = 0.1; // Integration step size [h]
    let mut t: f64 = 0.0;

    while t < target_time {
        let step = dt.min(target_time - t);
        state = rk4_step(state, step, ka, cl, v);
        t += step;
    }

    // Return concentration [mg/L]
    state.a_central / v
}

/// Simple pseudo-random number generator (LCG)
/// Used to avoid external dependencies
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        // Linear congruential generator
        const A: u64 = 6364136223846793005;
        const C: u64 = 1442695040888963407;
        self.state = self.state.wrapping_mul(A).wrapping_add(C);
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        // Generate uniform [0, 1)
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        // Box-Muller transform for standard normal
        let u1 = self.next_f64();
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }

    fn uniform(&mut self, min: f64, max: f64) -> f64 {
        min + (max - min) * self.next_f64()
    }
}

/// Generate synthetic dataset
pub fn generate_dataset(
    n_subjects: usize,
    obs_times: &[f64],
    dose_amount: f64,
    params: &TrueParams,
    seed: u64,
) -> Vec<DataRow> {
    let mut rng = SimpleRng::new(seed);
    let mut rows = Vec::new();

    for subject_id in 1..=n_subjects {
        // Sample body weight [50-90 kg]
        let wt = rng.uniform(50.0, 90.0);

        // Sample random effects (IIV)
        let eta_cl = rng.normal() * params.omega_cl;
        let eta_v = rng.normal() * params.omega_v;
        let eta_ka = rng.normal() * params.omega_ka;

        // Compute individual parameters with allometric scaling
        let w_norm = wt / 70.0;
        let cl_i = params.cl_pop * w_norm.powf(0.75) * eta_cl.exp();
        let v_i = params.v_pop * w_norm * eta_v.exp();
        let ka_i = params.ka_pop * eta_ka.exp();

        // Dose row (EVID=1)
        rows.push(DataRow {
            id: subject_id,
            time: 0.0,
            dv: None,
            wt,
            evid: 1,
            amt: Some(dose_amount),
        });

        // Observation rows (EVID=0)
        for &time in obs_times {
            let c_pred = solve_pk(ka_i, cl_i, v_i, dose_amount, time);

            // Add proportional error
            let epsilon = rng.normal() * params.sigma_prop;
            let dv = (c_pred * (1.0 + epsilon)).max(0.0); // Truncate at 0

            rows.push(DataRow {
                id: subject_id,
                time,
                dv: Some(dv),
                wt,
                evid: 0,
                amt: None,
            });
        }
    }

    rows
}

/// Write dataset to CSV file
pub fn write_csv(path: &str, data: &[DataRow]) -> std::io::Result<()> {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create(path)?;

    // Header
    writeln!(file, "ID,TIME,DV,WT,EVID,AMT")?;

    // Rows
    for row in data {
        write!(file, "{},{:.6},", row.id, row.time)?;

        // DV column
        match row.dv {
            Some(dv) => write!(file, "{:.6}", dv)?,
            None => write!(file, "NA")?,
        }

        write!(file, ",{:.6},{},", row.wt, row.evid)?;

        // AMT column
        match row.amt {
            Some(amt) => writeln!(file, "{:.6}", amt)?,
            None => writeln!(file, "NA")?,
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rng_reproducible() {
        let mut rng1 = SimpleRng::new(42);
        let mut rng2 = SimpleRng::new(42);

        for _ in 0..100 {
            assert_eq!(rng1.next_f64(), rng2.next_f64());
        }
    }

    #[test]
    fn test_solve_pk_steady_state() {
        // At very long times, central compartment should approach 0
        let c = solve_pk(1.0, 10.0, 50.0, 100.0, 100.0);
        assert!(c < 0.01); // Should be near zero
    }

    #[test]
    fn test_generate_dataset() {
        let params = TrueParams::default();
        let obs_times = vec![1.0, 2.0, 4.0, 8.0];
        let data = generate_dataset(5, &obs_times, 100.0, &params, 12345);

        // Should have 5 dose rows + 5*4 observation rows = 25 total
        assert_eq!(data.len(), 5 + 5 * 4);

        // Check first row is a dose
        assert_eq!(data[0].evid, 1);
        assert!(data[0].amt.is_some());
        assert!(data[0].dv.is_none());

        // Check second row is an observation
        assert_eq!(data[1].evid, 0);
        assert!(data[1].amt.is_none());
        assert!(data[1].dv.is_some());
    }
}
