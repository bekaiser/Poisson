# Derivatives, spectral anti-derivatives, and de-aliasing by discrete sine 
# transforms in 1D. 
# Bryan Kaiser 
# 4/8/17

using DataArrays
using PyPlot
using PyCall
using Base.FFTW

# add derivative function
# add way to deal with mean

# =============================================================================
# readme:

# This script shows how to take 1D derivatives, do 1D spectral inversions, 
# and how to de-alias nonlinear signals using FFTW DSTs. 

# Make sure that you create the directory "/figures" in the same directory as 
# as this script, for output plots.

# =============================================================================
# functions

function dealias(U,V,ks)
	# dst de-aliasing
	for j = 1:length(ks)
		if abs(ks[j]) >= length(U)/3.0*ks[1]; # 2/3 rule
			U[j] = 0.0; V[j] = 0.0; # Fourier space variables
		end
	end
	uv = r2r(U,FFTW.RODFT01).*r2r(V,FFTW.RODFT01); # de-aliased product
	return uv
end

function poisson(q,k)
	# 1D DST spectral inversion using FFTW to solve the Poisson equation:
	# given q and the Poisson equation, Laplacian(psi)=q, solve for psi.
	psi = r2r(-r2r(q,FFTW.RODFT10)./(2.0*length(q).*k.^(2.0)),FFTW.RODFT01);
	return psi 
end

function bandavg(U::Array{Float64,1},nb::Int64)
	# nb = number of bands to average over (window size)
	# U = spectral variable to be band-averaged, low to high frequency.
	N = length(U);
	Nb = Int64(floor(N/nb)); Ub = zeros(Nb);
	r = Int64(N-Nb*nb); # remainder 
	#println("$Nb,$N,$r,$(length(collect(1:Nb)))")
	for j = 1:Nb
		#println("$(length(collect((r+1+(j-1)*nb):(r+j*nb))))")
		Ub[j] = sum(U[(r+1+(j-1)*nb):(r+j*nb)])/Float64(nb);
	end
	return Ub
end

# =============================================================================
# domain
 
L = 3000.0 # km, domain size
Lcenter = 0.0 # x value @ the center of the grid
N = 2^8 # series length (must be at least even)
dx = L/Float64(N) # km, uniform longitudinal grid spacing
x = collect(0.5*dx:dx:dx*Float64(N))-(L/2.0-Lcenter) # km, centered uniform grid 


# =============================================================================
# test signals

ifield = 2

if ifield == 1 # cos(kx), problematic for a sine transform to resolve
A = 3.0 # m/s, velocity signal amplitude
lambda = L/10.0 # km, wavelength of signal (smallest possible)
ks = 2.0*pi/lambda # rad/km
u = A*cos(ks*x) # km/s (not physical...)
dudxA = -A*ks*sin(ks*x) 
d2udx2A = -A*ks^2.0*cos(ks*x) 
elseif ifield == 2 # sin(kx), machine precision even derivatives
A = 3.0 # m/s, velocity signal amplitude
lambda = L/2.0 # km, wavelength of signal (smallest possible)
ks = 2.0*pi/lambda # rad/km
u = A*sin(ks*x) # km/s
dudxA = A*ks*cos(ks*x) 
d2udx2A = -A*ks^2.0*sin(ks*x)
elseif ifield == 3 # Gaussian(x)
sigma = L/20.0
X = x-Lxcenter
u = exp(-X.^2.0/(2.0*sigma^2.0)) # test signal equivalent to gaussmf(x,[L/10 0]);
dudxA = X.*u.*(-sigma^(-2.0)) # test signal derivative (Gaussian)
d2udx2A = u.*(X.^2.0-sigma^2.0)./sigma^4.0;
elseif ifield == 4
m = 2.0; b = L; u = x.*m+b;
dudxA = ones(size(x)).*m;
d2udx2A = zeros(size(x));
end
 
# signal plot 
fig = figure(); CP = plot(x./L,u,"k");
xlabel("x"); ylabel("u"); title("signal");
savefig("./figures/signal.png",format="png"); close(fig);


# =============================================================================
# sine/cosine transforms

Ucos = r2r(u,FFTW.REDFT10)./(2.0*Float64(N)); # DCT-II 
Usin = r2r(u,FFTW.RODFT10)./(2.0*Float64(N)); # DST-II 

#PUsin = plan_r2r(flags=FFTW.MEASURE,u,FFTW.RODFT10)./(2.0*Float64(N))
#UsinP = PUsin*u;

# wavenumbers 
kc = collect(0:Int64(N)-1).*(pi/L); # for DCT-II
ks = collect(1:Int64(N)).*(pi/L); # for DST-II

hs = ks./pi; # 1/length, equivalent to Hz for time

# comparison of transforms
fig = figure(); plot(kc,Ucos,"r",label="DCT-II"); 
plot(ks,Usin,"b",label="DST-II");
xlabel("k (rads/length)"); title("transforms"); legend();
savefig("./figures/spectral_dst_dct.png",format="png"); close(fig);


# =============================================================================
# inverse discrete sine transform

uinv = r2r(Usin,FFTW.RODFT01); # DST-III (IDST)

# comparison of time series and reconstructed time series 
fig = figure(); plot(x./L,u,"k",label="signal");
plot(x./L,uinv,"b",label="IDST");
xlabel("x"); title("DST-III (IDST) of DST-II transform"); legend();
savefig("./figures/reconstructed_time_series.png",format="png"); close(fig);

# DST => IDST error plot
fig = figure(); CP = semilogy(x./L,abs(u-real(uinv)),"k");
xlabel("x"); title("DST-III (IDST) of DST-II transform, error");
savefig("./figures/reconstructed_time_series_error.png",format="png"); close(fig);


# =============================================================================
# first derivative

Ushift = [0.0;Usin[1:Int64(N)-1]]; # sine shift (pi/2) for 1st derivative

# plot of transformed derivative
fig = figure(); plot(ks,Usin,"k",label="sine transformed signal");
plot(kc,-Ushift,"b",label="transformed derivative");
xlabel("k (rads/length)"); title("du/dx transform"); legend();
savefig("./figures/spectral_first_derivative.png",format="png"); close(fig);

# signal derivative via cosine transform 
dudx = r2r(Ushift.*kc,FFTW.REDFT01); # DCT-III (IDCT) 

# comparison of analytical du/dx and computed du/dx plot
fig = figure(); plot(x./L,dudxA,"r",label="signal");
plot(x./L,dudx,"b--",label="IDCT"); xlabel("x"); 
title("du/dx by DCT-III (IDCT) of shifted DST-II"); legend();
savefig("./figures/first_derivative.png",format="png"); close(fig);

deriv_error = abs(dudxA-dudx); max_deriv_error = maximum(abs(dudxA-dudx));
println("\nThe maximum first derivative error is $(max_deriv_error) for $N grid points\n");

# first derivative error plot
fig = figure(); CP = semilogy(x./L,deriv_error,"k");
xlabel("x"); title("du/dx by DCT-III (IDCT) of shifted DST-II, error");
savefig("./figures/first_derivative_error.png",format="png"); close(fig);


# =============================================================================
# second derivative

d2udx2 = r2r(-Usin.*ks.^2.0,FFTW.RODFT01); # DST-III

# plot of transformed derivative
fig = figure(); plot(ks,Usin,"k",label="sine transformed signal");
plot(ks,-Usin.*ks.^2.0,"b",label="transformed derivative"); 
xlabel("k (rads/length)"); title("d^2u/dx^2 transform"); legend();
savefig("./figures/spectral_second_derivative.png",format="png"); close(fig);

# comparison with analytical d^2u/dx^2
fig = figure(); plot(x./L,d2udx2A,"k",label="signal");
plot(x./L,d2udx2,"b",label="IDST"); 
xlabel("x"); title("d^2u/dx^2 by DST-III (IDST) of DST-II"); legend();
savefig("./figures/second_derivative.png",format="png"); close(fig);

deriv2_error = abs(d2udx2A-d2udx2); max_deriv2_error = maximum(abs(d2udx2A-d2udx2));
println("The maximum second derivative error is $(max_deriv2_error) for $N grid points\n");

# error plot
fig = figure(); semilogy(x,deriv2_error,"k");
xlabel("x"); title("d^2u/dx^2 by DST-III (IDST) of DST-II, error");
savefig("./figures/second_derivative_error.png",format="png"); close(fig);


# =============================================================================
# de-aliasing

alias_signal = 1;

if alias_signal == 1 
# random signals
A = 1.0; # m/s, velocity signal amplitude
ua = rand(size(x)).*A; ub = rand(size(x)).*A;
elseif alias_signal == 2
# harmonic signals
ka = Float64(N)/100.0*(2.0*pi/L);
ua = sin(x.*ka); ub = u1;
end

Ua = r2r(ua,FFTW.RODFT10)./(2.0*Float64(N)); # DST-II 
Ub = r2r(ub,FFTW.RODFT10)./(2.0*Float64(N)); # DST-II 

# random signal plot 
fig = figure(); plot(x./L,ua,"r",label="u_1");
plot(x./L,ub,"b",label="u_2");
xlabel("x"); legend(); title("random signals");
savefig("./figures/random_signals.png",format="png"); close(fig);

# band-averaging
Nb = 4; hsb = bandavg(hs,Nb); 
Uab = bandavg(Ua,Nb); Ubb = bandavg(Ub,Nb);

# single-sided spectrum plot
fig = figure(); 
semilogx(hs,2.0/Float64(N)*abs(Ua),"r",label="u_1");
semilogx(hs,2.0/Float64(N)*abs(Ub),"b",label="u_2"); 
semilogx(hsb,2.0/Float64(N)*abs(Uab),"r--",label="u_1 band");
semilogx(hsb,2.0/Float64(N)*abs(Ubb),"b--",label="u_2 band");
ylabel("|U|");
xlabel("spatial frequency (1/length)"); legend(loc=2); title("Single sided spectrum");
savefig("./figures/single_sided_white_spectrums.png",format="png"); close(fig);

u2_alias = ua.*ub; # aliased square
u2_dealias = dealias(Ua,Ub,ks); # de-aliased square
U2_alias = r2r(u2_alias,FFTW.RODFT10)./(2.0*Float64(N)); # DST-II 
U2_dealias = r2r(u2_dealias,FFTW.RODFT10)./(2.0*Float64(N)); # DST-II 

# alias vs. de-aliased physical plot 
fig = figure(); plot(x./L,u2_alias,"r",label="aliased");
plot(x./L,u2_dealias,"b",label="de-aliased");
xlabel("x"); legend(); title("u_1*u_2");
savefig("./figures/aliased_signals.png",format="png"); close(fig);

# single-sided spectrum plot
fig = figure(); semilogx(hs,2.0/float(N)*abs(U2_alias),"r",label="aliased");
semilogx(hs,2.0/float(N)*abs(U2_dealias),"b",label="de-aliased"); ylabel("|U^2|");
xlabel("spatial frequency (1/length)"); legend(loc=2); title("u_1*u_2");
#axis([pi/(4.0*Lx),float(N)*pi/(5.0*Lx),0.0,0.05]);
savefig("./figures/aliasing_spectrums.png",format="png"); close(fig);


# =============================================================================
# Poisson equation solution by dst

# Poisson equation solution: Laplacian(psi) = qA
uP = poisson(d2udx2A,ks); 
Poisson_error = abs(uP-u); # Poisson equation solution error
max_err = maximum(Poisson_error);

fig = figure(); plot(x./L,u,"k",label="signal"); 
plot(x./L,uP,"b",label="Poisson solution"); legend();
xlabel("x"); ylabel("u"); title("Laplacian(psi) = q, psi solution"); 
savefig("./figures/Poisson_solution.png",format="png"); close(fig);

fig = figure(); CP = plot(x./L,Poisson_error,"k")
xlabel("x"); ylabel("error"); title("Laplacian(psi) = q, psi solution error"); 
savefig("./figures/Poisson_solution_error.png",format="png"); close(fig);

println("The maximum Poisson equation computation error is $(max_err) for $N gridpoints.\n")


# =============================================================================
# fftw plan

n = collect(3:23); # typeof(n) = Array{Int64,1}, powers of 2 grid resolution

# initialized fields
dst_time = zeros(length(n)); dst_plan_time = zeros(length(n));
Linf_d1 = zeros(length(n)); Linf_d2 = zeros(length(n));
Linf_d1P = zeros(length(n)); Linf_d2P = zeros(length(n));

for m = 1:length(n) # loop over powers of 2 grid resolution 

# domain
L = 3000.0 # km, domain size
Lcenter = 0.0 # x value @ the center of the grid
N = 2^n[m] # series length (must be at least even)
dx = L/Float64(N) # km, uniform longitudinal grid spacing
x = collect(0.5*dx:dx:dx*Float64(N))-(L/2.0-Lcenter) # km, centered uniform grid 

# signal
if ifield == 1 # cos(kx), problematic for a sine transform to resolve
A = 3.0 # m/s, velocity signal amplitude
lambda = L/10.0 # km, wavelength of signal (smallest possible)
ks = 2.0*pi/lambda # rad/km
u = A*cos(ks*x) # km/s (not physical...)
dudxA = -A*ks*sin(ks*x) 
d2udx2A = -A*ks^2.0*cos(ks*x) 
elseif ifield == 2 # sin(kx), machine precision even derivatives
A = 3.0 # m/s, velocity signal amplitude
lambda = L/2.0 # km, wavelength of signal (smallest possible)
ks = 2.0*pi/lambda # rad/km
u = A*sin(ks*x) # km/s
dudxA = A*ks*cos(ks*x) 
d2udx2A = -A*ks^2.0*sin(ks*x)
elseif ifield == 3 # Gaussian(x)
sigma = L/20.0
X = x-Lxcenter
u = exp(-X.^2.0/(2.0*sigma^2.0)) # test signal equivalent to gaussmf(x,[L/10 0]);
dudxA = X.*u.*(-sigma^(-2.0)) # test signal derivative (Gaussian)
d2udx2A = u.*(X.^2.0-sigma^2.0)./sigma^4.0;
elseif ifield == 4
m = 2.0; b = L; u = x.*m+b;
dudxA = ones(size(x)).*m;
d2udx2A = zeros(size(x));
end

# wavenumbers 
kc = collect(0:Int32(N)-1).*(pi/L); # for DCT-II
ks = collect(1:Int32(N)).*(pi/L); # for DST-II

# initialize spectral u, derivatives
Usin = similar(x); Ushift = similar(x); dudx = similar(x); d2udx2 = similar(x);
UsinP = similar(x); UshiftP = similar(x); dudxP = similar(x); d2udx2P = similar(x);

tic(); for m1 = 1:100;
#Ucos = r2r(u,FFTW.REDFT10)./(2.0*Float64(N)); # DCT-II 
Usin = r2r(u,FFTW.RODFT10)./(2.0*Float64(N)); # DST-II 
Ushift = [0.0;Usin[1:Int64(N)-1]]; # sine shift (pi/2) for 1st derivative
dudx = r2r(Ushift.*kc,FFTW.REDFT01); # DCT-III (IDCT) 
d2udx2 = r2r(-Usin.*ks.^2.0,FFTW.RODFT01); # DST-III
end; dst_time[m] = toq();

# multithread number of threads and fft_plan:
FFTW.set_num_threads(4)
PUcos = plan_r2r(u,FFTW.REDFT10,flags=FFTW.MEASURE);
PUsin = plan_r2r(u,FFTW.RODFT10,flags=FFTW.MEASURE); 
#UcosP = (PUcos*u)./(2.0*Float64(N)); 
UsinP = (PUsin*u)./(2.0*Float64(N)); 
UshiftP = [0.0;UsinP[1:Int64(N)-1]]; # sine shift (pi/2) for 1st derivative
Pdudx = plan_r2r(UshiftP.*kc,FFTW.REDFT01,flags=FFTW.MEASURE);
Pd2udx2 = plan_r2r(-UsinP.*ks.^2.0,FFTW.RODFT01,flags=FFTW.MEASURE)
dudxP = Pdudx*(UshiftP.*kc); 
d2udx2P = Pd2udx2*(-Usin.*ks.^2.0); 

tic(); for m2 = 1:100; 
#UcosP = (PUcos*u)./(2.0*Float64(N)); 
UsinP = (PUsin*u)./(2.0*Float64(N)); 
UshiftP = [0.0;UsinP[1:Int64(N)-1]]; # sine shift (pi/2) for 1st derivative
dudxP = Pdudx*(UshiftP.*kc); 
d2udx2P = Pd2udx2*(-UsinP.*ks.^2.0); 
end; dst_plan_time[m] = toq();

# infinity norm error for fft and fft plan 
Linf_d1[m] = maximum(abs(dudxA-dudx));
Linf_d2[m] = maximum(abs(d2udx2A-d2udx2));
Linf_d1P[m] = maximum(abs(dudxA-dudxP));
Linf_d2P[m] = maximum(abs(d2udx2A-d2udx2P));

#=
deriv_error = abs(dudxA-dudx); max_deriv_error = maximum(abs(dudxP-dudx));
println("The maximum plan fftw first derivative error is $(max_deriv_error) for $N grid points\n");
deriv2_error = abs(d2udx2A-d2udx2); max_deriv2_error = maximum(abs(d2udx2P-d2udx2));
println("The maximum second derivative error is $(max_deriv2_error) for $N grid points\n");
=#

end

Nvec = (ones(size(n)).*2.0).^n;
NlogN = (log2(Nvec).*Nvec).*(dst_time[1]/(log2(Nvec[1]).*Nvec[1]));

# wall time plot
fig = figure(); 
loglog(Nvec,NlogN,"k--",label="Nlog2(N)");
loglog(Nvec,dst_time,"k",label="dst"); 
loglog(Nvec,dst_plan_time,"r--",label="plan_dst"); legend(loc=2);
xlabel("N"); ylabel("seconds"); title("wall time, 1000 dsts"); 
savefig("./figures/dst_plan_computation_time.png",format="png"); close(fig);

# error plot
fig = figure(); loglog(Nvec,Linf_d1,"b--",label="dx, dst"); 
loglog(Nvec,Linf_d2,"b",label="dx2, dst"); 
loglog(Nvec,Linf_d1P,"r.",label="dx, dst plan"); 
loglog(Nvec,Linf_d2P,"r--",label="dx2, dst plan"); legend(loc=4);
xlabel("N"); ylabel("L infinity"); title("error"); 
savefig("./figures/dst_plan_error.png",format="png"); close(fig);
