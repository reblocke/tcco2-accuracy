data {
  int<lower=1> N;
  int<lower=0,upper=N> I;  // independent summaries first
  vector[N] bias;
  vector[N] ell;
  vector<lower=0>[I] v_bias;
  vector<lower=0>[I] v_ell;
  vector[I] cov_bias_ell;
  int<lower=0> B;
  array[B] int<lower=2> block_size;
  matrix[2*(N-I),2*(N-I)] V_dep;
  array[N-I] int<lower=1> setting_id;
  real prior_mu_b;
  real<lower=0> prior_mu_b_sd;
  real prior_mu_ell;
  real<lower=0> prior_mu_ell_sd;
  real<lower=0> prior_tau_b;
  real<lower=0> prior_tau_ell;
  real<lower=0> eta;
}
transformed data {
  if (sum(block_size) != N-I) reject("dependent block sizes disagree");
}
parameters {
  real mu_b;
  real mu_ell;
  real<lower=0> tau_b;
  real<lower=0> tau_ell;
  real<lower=-1,upper=1> rho;
}
model {
  real cov_true = rho*tau_b*tau_ell;
  int start = 1;
  mu_b ~ normal(prior_mu_b, prior_mu_b_sd);
  mu_ell ~ normal(prior_mu_ell, prior_mu_ell_sd);
  tau_b ~ normal(0, prior_tau_b);
  tau_ell ~ normal(0, prior_tau_ell);
  // The two-dimensional LKJ density and its normalizing constant.
  target += beta_lpdf((rho+1)/2 | eta,eta) - log(2);
  if (I > 0) {
    vector[I] vb = v_bias + square(tau_b);
    vector[I] c = cov_bias_ell + cov_true;
    head(bias,I) ~ normal(mu_b,sqrt(vb));
    head(ell,I) ~ normal(mu_ell + c ./ vb .* (head(bias,I)-mu_b),
                         sqrt(v_ell + square(tau_ell) - square(c) ./ vb));
  }
  for (b in 1:B) {
    int k = block_size[b];
    vector[2*k] residual;
    matrix[2*k,2*k] cov = block(V_dep,2*start-1,2*start-1,2*k,2*k);
    for (j in 1:k) {
      int id = start+j-1;
      residual[2*j-1] = bias[I+id]-mu_b;
      residual[2*j] = ell[I+id]-mu_ell;
      for (h in 1:k) {
        if (setting_id[id] == setting_id[start+h-1]) {
          cov[2*j-1,2*h-1] += square(tau_b);
          cov[2*j,2*h] += square(tau_ell);
          cov[2*j-1,2*h] += cov_true;
          cov[2*j,2*h-1] += cov_true;
        }
      }
    }
    residual ~ multi_normal_cholesky(rep_vector(0,2*k),cholesky_decompose(cov));
    start += k;
  }
}
