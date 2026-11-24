# Run with Rscript --vanilla overall-envelope.R /path/to/data.Rdata
# Input is Conway et al.'s Figshare v2 data.Rdata (file 11409167, CC BY 4.0).
# Compare the archive SHA-256 with models/future_setting/specification.json.
# Supplementary Table 2 uses Tipton and Shuster, equations 4.13–4.16.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Supply the path to the checked Figshare v2 data.Rdata file.")
records <- new.env()
load(args[1], envir = records)
tab <- records$main
k <- 75L
needed <- c("bias", "V_bias", "logs2", "V_logs2", "S2*", "n")
stopifnot(is.data.frame(tab), nrow(tab) == k, all(needed %in% names(tab)))
stopifnot(all(is.finite(as.matrix(tab[needed]))), all(tab[["S2*"]] > 0), all(tab$n > 1))
stopifnot(isTRUE(all.equal(tab$logs2, log10(tab[["S2*"]]) + 1 / (tab$n - 1))))
stopifnot(isTRUE(all.equal(tab$V_logs2, 2 / (tab$n - 1))))

pool <- function(effect, sampling_var) {
    fixed_weights <- 1 / sampling_var
    fixed_mean <- weighted.mean(effect, fixed_weights)
    q <- sum(fixed_weights * (effect - fixed_mean)^2)
    denominator <- sum(fixed_weights) - sum(fixed_weights^2) / sum(fixed_weights)
    tau2 <- (q - (k - 1)) / denominator
    weights <- 1 / (sampling_var + tau2)
    center <- weighted.mean(effect, weights)
    robust_var <- (k / (k - 1)) * sum((weights * (effect - center))^2) / sum(weights)^2
    c(center = center, tau2 = tau2, robust_var = robust_var)
}

bias <- pool(tab$bias, tab$V_bias)
tau2 <- unname(bias["tau2"])
var_tau2 <- 2 / sum((tab$V_bias + tau2)^(-2))
t_critical <- qt(0.975, df = k - 1)
natural_log <- log(tab[["S2*"]]) + 1 / (tab$n - 1)

compare <- function(label, log_var, direct_scale) {
    within <- pool(log_var, tab$V_logs2)
    sigma2 <- exp(unname(within["center"]))
    total <- sigma2 + tau2
    limits <- unname(bias["center"]) + c(-2, 2) * sqrt(total)
    bias_term <- unname(bias["robust_var"])
    within_term <- sigma2^2 * unname(within["robust_var"]) / total
    tau_term <- var_tau2 / total
    if (!direct_scale) tau_term <- tau2^2 * tau_term
    endpoint_se <- sqrt(bias_term + within_term + tau_term)
    margin <- t_critical * endpoint_se
    data.frame(
        calculation = label, bias = unname(bias["center"]), within_sd = sqrt(sigma2),
        tau2 = tau2, var_tau2 = var_tau2, lower = limits[1], upper = limits[2],
        outer_lower = limits[1] - margin, outer_upper = limits[2] + margin,
        bias_term = bias_term, within_term = within_term, tau_term = tau_term,
        endpoint_se = endpoint_se, t_critical = t_critical, margin = margin
    )
}

results <- rbind(
    compare("Archived calculation", tab$logs2, FALSE),
    compare("Natural log only", natural_log, FALSE),
    compare("Endpoint variance only", tab$logs2, TRUE),
    compare("Both corrections", natural_log, TRUE)
)
write.table(results, row.names = FALSE, sep = "\t", quote = FALSE)
