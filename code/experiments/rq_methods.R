# R baselines for P(sum_i theta_i chi2_1 <= y) on identical weights (E5, revised 17 Nov 2026).
# Usage: Rscript rq_methods.R <weights.csv> <y> <out.csv> [reps]
#   weights.csv: one column 'theta' (exactly one positive weight for the ratio statistics of this paper)
# Direct methods on the indefinite form: Davies (1980, AS 155) and Imhof (1961) from CompQuadForm.
# Liu-Tang-Zhang (2009, CompQuadForm::liu) is documented for non-negative definite forms only; it is run and labelled
# "outside documented domain", not used in the comparison.
# Conditional methods: HBE, LPB4 (momentchi2) and the Satterthwaite / saddlepoint options of survey::pchisqsum are applied
# to the NEGATIVE part W = sum a_i chi2_1 (a_i = -theta_i > 0, in their domain) and integrated over the positive part,
#   P(X <= y) = E_Z[ 1 - F_W(theta_+ Z^2 - y) ] = 2 * [ Phi(z0) - 1/2 + int_{z0}^inf (1 - F_W(theta_+ z^2 - y)) phi(z) dz ],
#   z0 = sqrt(max(y,0)/theta_+): adaptive quadrature (stats::integrate, rel.tol 1e-11) starting at the kink z0, on
#   [z0, z0 + 40] (omitted mass < 2 pnorm(-40) < 1e-300; the saddlepoint option of pchisqsum fails at u ~ 1e6).
#   The former 64-node Gauss-Hermite rule is kept as a diagnostic ("GH-64") to show the outer-quadrature error.
suppressMessages({library(CompQuadForm); library(momentchi2); library(survey)})
args <- commandArgs(trailingOnly = TRUE)
th <- read.csv(args[1])$theta
y <- as.numeric(args[2])
reps <- if (length(args) >= 4) as.integer(args[4]) else 3L
tp <- max(th); a <- -th[th < 0]
z0 <- sqrt(max(y, 0) / tp)
gh <- function(n) { i <- 1:(n - 1); J <- matrix(0, n, n); J[cbind(i, i + 1)] <- sqrt(i / 2); J[cbind(i + 1, i)] <- sqrt(i / 2)
  e <- eigen(J, symmetric = TRUE); list(x = e$values * sqrt(2), w = e$vectors[1, ]^2) }
G <- gh(64)
cond_gh <- function(FW) { v <- 0; for (j in seq_along(G$x)) { u <- tp * G$x[j]^2 - y; v <- v + G$w[j] * (if (u <= 0) 1 else 1 - FW(u)) }; v }
cond_ad <- function(FW) {
  # outer integral on [z0, z0 + 40], split at z0 + 8; the omitted mass is below 2 * pnorm(-40) < 1e-300
  f <- function(z) sapply(z, function(zz) { u <- tp * zz^2 - y; if (u <= 0) dnorm(zz) else (1 - FW(u)) * dnorm(zz) })
  I1 <- integrate(f, z0, z0 + 8, rel.tol = 1e-11, abs.tol = 1e-16, subdivisions = 2000L, stop.on.error = FALSE)
  I2 <- integrate(f, z0 + 8, z0 + 40, rel.tol = 1e-11, abs.tol = 1e-30, subdivisions = 2000L, stop.on.error = FALSE)
  v <- 2 * (pnorm(z0) - 0.5 + I1$value + I2$value)
  msg <- unique(c(I1$message, I2$message)); msg <- msg[msg != "OK"]
  attr(v, "msg") <- if (length(msg)) paste0("integrate: ", paste(msg, collapse = "; "), " (value returned; abs.error ",
                                             signif(2 * (I1$abs.error + I2$abs.error), 2), ")") else ""
  v
}
# per-call wall time: one warm-up call, then for each repetition the call is repeated until >= 0.2 s have elapsed
timeit <- function(f) {
  val <- f(); tt <- numeric(reps)
  for (r in 1:reps) { n <- 1L
    repeat { t0 <- Sys.time(); for (i in seq_len(n)) val <- f(); el <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
             if (el >= 0.2 || n >= 100000L) break; n <- n * 4L }
    tt[r] <- el / n }
  list(val = val, sec = min(tt), sec_med = median(tt)) }
res <- list()
add <- function(name, f, note = "") {
  out <- tryCatch(timeit(f), error = function(e) list(val = NA, sec = NA, sec_med = NA, err = conditionMessage(e)))
  m <- if (is.null(out$err)) { a <- attr(out$val, "msg"); if (!is.null(a) && nchar(a)) paste(c(note[nchar(note) > 0], a), collapse = "; ") else note } else out$err
  res[[length(res) + 1]] <<- data.frame(method = name, value = as.numeric(out$val), seconds = out$sec, seconds_median = out$sec_med,
                                        note = m, stringsAsFactors = FALSE)
}
for (acc in c(1e-10, 1e-8, 1e-6)) {
  d <- davies(y, th, acc = acc, lim = 200000)
  if (d$ifault == 0) { add(sprintf("R CompQuadForm::davies (acc %g)", acc), function() 1 - davies(y, th, acc = acc, lim = 200000)$Qq); break }
}
add("R CompQuadForm::imhof", function() 1 - imhof(y, th, epsabs = 1e-12, epsrel = 1e-12, limit = 20000)$Qq)
add("R CompQuadForm::liu (LTZ 2009)", function() 1 - liu(y, th), note = "outside documented domain (indefinite form)")
for (m in list(list("hbe", function(u) hbe(a, u)), list("lpb4", function(u) lpb4(a, u)),
               list("pchisqsum saddlepoint", function(u) pchisqsum(u, rep(1, length(a)), a, method = "saddlepoint")),
               list("pchisqsum satterthwaite", function(u) pchisqsum(u, rep(1, length(a)), a, method = "satterthwaite")))) {
  pkg <- if (m[[1]] %in% c("hbe", "lpb4")) "momentchi2" else "survey"
  add(sprintf("R %s::%s | adaptive over Z", pkg, m[[1]]), function() cond_ad(m[[2]]))
  add(sprintf("R %s::%s | GH-64 over Z (diagnostic)", pkg, m[[1]]), function() cond_gh(m[[2]]), note = "outer-quadrature diagnostic")
}
write.csv(do.call(rbind, res), args[3], row.names = FALSE)
