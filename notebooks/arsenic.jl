### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ 8183b318-5ebb-11eb-1cd8-a96e8704a378
using Pkg, DrWatson

# ╔═╡ 686dac30-5ebb-11eb-00f1-434980dba906
begin
	@quickactivate "ParetoSmoothedImportanceSamplng"
	using ParetoSmoothedImportanceSampling
	using StanSample, StatsFuns, StatsPlots
	using DataFrames, CSV, JSON
end

# ╔═╡ 923212a8-630d-11eb-390f-75d21be80011
begin
	ProjDir = joinpath(psis_path, "..", "examples", "arsenic")
	include(joinpath(ProjDir, "cvit.jl"))
end;

# ╔═╡ ca0d916e-5ebe-11eb-3af1-0993bf5b82c8
md" ### Psis_loo."

# ╔═╡ 2ee71b16-5ebd-11eb-3a7d-4553f2f22274
md" ##### Data."

# ╔═╡ 37f06bc2-5ebd-11eb-31f6-65bfa74ea9cf
begin
	data = JSON.parsefile(joinpath(ProjDir, "wells.data.json"))
	y = Float64.(data["switched"])
	x = Float64[data["arsenic"]  data["dist"]]
	n, m = size(x)
end

# ╔═╡ 37f0b87a-5ebd-11eb-092a-3d9b9e8e427b
md" ##### Model."

# ╔═╡ 37f14af6-5ebd-11eb-0aae-d57e8b3d951a
begin
	model_str = read(open(joinpath(ProjDir, "arsenic_logistic.stan")), String)
	tmpdir = joinpath(ProjDir, "tmp")
	sm1 = SampleModel("arsenic_logistic", model_str; tmpdir)
	data1 = (p = m, N = n, y = Int.(y), x = x)
	rc1 = stan_sample(sm1; data=data1)
end

# ╔═╡ 38128be4-5ebd-11eb-36da-a96a68ca1316
md" ##### Fit the model in Stan."

# ╔═╡ 38135bc8-5ebd-11eb-0644-511bb08b3e59
if success(rc1)
	nt1 = read_samples(sm1)

	# Compute LOO and standard error
	
	log_lik1 = nt1.log_lik'
	loo1, loos1, pk1 = psisloo(log_lik1)
	elpd_loo1 = sum(loos1)
	se_elpd_loo1 = std(loos1) * sqrt(n)
	(elpd_loo = elpd_loo1, se_elpd_loo = se_elpd_loo1)
end

# ╔═╡ 44457a20-5ebd-11eb-00a1-855bfbb45e22
if success(rc1)
	
	# Check the shape parameter k of the generalized Pareto distribution
	
	pk_qualify(pk1)
end

# ╔═╡ 5a76e4aa-5ebd-11eb-2e15-6d5808ead825
md" ##### Fit a second model, using log(arsenic) instead of arsenic."

# ╔═╡ 5a771bfa-5ebd-11eb-1431-a5bc45055a5a
begin
	x2 = Float64[log.(data["arsenic"])  data["dist"]]
	data2 = (p = m, N = n, y = Int.(y), x = x2)
	rc2 = stan_sample(sm1; data=data2)
end;

# ╔═╡ 5a77d188-5ebd-11eb-0090-8d80d495f1b1
if success(rc2)
    nt2 = read_samples(sm1)
	
    # Compute LOO and standard error
	
    log_lik2 = nt2.log_lik'
    loo2, loos2, pk2 = psisloo(log_lik2)
    elpd_loo2 = sum(loos2)
    se_elpd_loo2 = std(loos2) * sqrt(n)
    (elpd_loo = elpd_loo2, se_elpd_loo = se_elpd_loo2)
end

# ╔═╡ b88657e2-5ebd-11eb-343e-9d666eef86cc
if success(rc2)
	
	# Check the shape parameter k of the generalized Pareto distribution
	
	pk_qualify(pk2)
end

# ╔═╡ dd28d430-5ebd-11eb-1854-ab4c13e82c34
if success(rc1) && success(rc2)
	
    ## Compare the models
	
    loodiff = loos1 - loos2
    (elpd_diff = sum(loodiff), se_elpd_diff = std(loodiff) * sqrt(n))
end

# ╔═╡ 98d7208e-5ebd-11eb-3a64-e568c1ceebc0
md" ### k-fold-CV."

# ╔═╡ e86ee35c-5ebd-11eb-1932-ffbde638f11e
md" ##### k-fold-CV should be used if several pk>0.5, in this case it is not needed, but provided as an example."

# ╔═╡ e86f0ab2-5ebd-11eb-1a0b-6958db3a7435
begin
	model_str_2 = read(open(joinpath(ProjDir, "arsenic_logistic_t.stan")), String)
	sm3 = SampleModel("arsenic_logistic_t", model_str_2)
end;

# ╔═╡ d83916ba-6309-11eb-309b-474efacd9f85
md" ##### Be patient... 10 cross Validations."

# ╔═╡ e70f73aa-5eb8-11eb-1960-bf6731681898
begin
	cvitr, cvitst = cvit(n, 10, true)
	kfcvs = similar(loos1)
	for cvi in 1:10
		standatacv = (p = m, N = length(cvitr[cvi]), Nt = length(cvitst[cvi]),
			x = x[cvitr[cvi],:], y = Int.(y[cvitr[cvi]]),
			xt = x[cvitst[cvi],:], yt = Int.(y[cvitst[cvi]]))

		# Fit the model in Stan
		rc3 = stan_sample(sm3; data=standatacv)
		if success(rc3)
			nt3 = read_samples(sm3)
			# Compute LOO and standard error
			log_likt = nt3.log_likt'
        	local n_sam, n_obs = size(log_likt)
        	kfcvs[cvitst[cvi]] .=
            	reshape(logsumexp(log_likt .- log(n_sam), dims=1), n_obs)
		end
	end
end

# ╔═╡ 1bf8d5a2-5ebe-11eb-12b3-2501e2f9d288
begin
	
	# compare PSIS-LOO and k-fold-CV
	
	plot([-3.5, 0], [-3.5, 0], color=:red)
	scatter!(loos1, kfcvs, xlab = "PSIS-LOO", ylab = "10-fold-CV",
		leg=false, color=:darkblue)
end

# ╔═╡ Cell order:
# ╠═8183b318-5ebb-11eb-1cd8-a96e8704a378
# ╠═686dac30-5ebb-11eb-00f1-434980dba906
# ╠═923212a8-630d-11eb-390f-75d21be80011
# ╟─ca0d916e-5ebe-11eb-3af1-0993bf5b82c8
# ╟─2ee71b16-5ebd-11eb-3a7d-4553f2f22274
# ╠═37f06bc2-5ebd-11eb-31f6-65bfa74ea9cf
# ╟─37f0b87a-5ebd-11eb-092a-3d9b9e8e427b
# ╠═37f14af6-5ebd-11eb-0aae-d57e8b3d951a
# ╟─38128be4-5ebd-11eb-36da-a96a68ca1316
# ╠═38135bc8-5ebd-11eb-0644-511bb08b3e59
# ╠═44457a20-5ebd-11eb-00a1-855bfbb45e22
# ╟─5a76e4aa-5ebd-11eb-2e15-6d5808ead825
# ╠═5a771bfa-5ebd-11eb-1431-a5bc45055a5a
# ╠═5a77d188-5ebd-11eb-0090-8d80d495f1b1
# ╠═b88657e2-5ebd-11eb-343e-9d666eef86cc
# ╠═dd28d430-5ebd-11eb-1854-ab4c13e82c34
# ╟─98d7208e-5ebd-11eb-3a64-e568c1ceebc0
# ╟─e86ee35c-5ebd-11eb-1932-ffbde638f11e
# ╠═e86f0ab2-5ebd-11eb-1a0b-6958db3a7435
# ╟─d83916ba-6309-11eb-309b-474efacd9f85
# ╠═e70f73aa-5eb8-11eb-1960-bf6731681898
# ╠═1bf8d5a2-5ebe-11eb-12b3-2501e2f9d288
