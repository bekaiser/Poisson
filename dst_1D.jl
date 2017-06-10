# Derivatives, spectral anti-derivatives, and de-aliasing by discrete cosine 
# transforms in 1D. 
# Bryan Kaiser 
# 3/10/17

using DataArrays
using PyPlot
using PyCall
using Base.FFTW


# =============================================================================
# readme:

# This script shows how to take 1D derivatives, do 1D spectral inversions, 
# and how to de-alias nonlinear signals using FFTW DCTs. 

# Make sure that you create the directory "/figures" in the same directory as 
# as this script, for output plots.


# =============================================================================
# functions

function dealias(U,V,kc)
	# dct de-aliasing
	for j = 1:length(kc)
		if abs(kc[j]) >= length(U)/3.0*kc[2]; # 2/3 rule
			U[j] = 0.0; V[j] = 0.0; # Fourier space variables
		end
	end
	uv = r2r(U,FFTW.REDFT01).*r2r(V,FFTW.REDFT01); # de-aliased product
	return uv
end

function poisson(q,kcinv)
	# 1D DST spectral inversion using FFTW to solve the Poisson equation:
	# given q and the Poisson equation, Laplacian(psi)=q, solve for psi.
	psi = r2r(-r2r(q,FFTW.REDFT10)./(2.0*length(q).*kcinv.^(2.0)),FFTW.REDFT01);
	return psi
end

function dct_d1(Ucos,ks) 
	# cosine shift (pi/2) for 1st derivative, then DST-III (inverse dst)
	dudx = r2r(-[Ucos[2:length(Ucos)];0.0].*ks,FFTW.RODFT01);
	return dudx
end

function dct_d2(Ucos,kc) 
	d2udx2 = r2r(-Ucos.*kc.^2.0,FFTW.REDFT01) # DCT-III;
	return d2udx2
end

# =============================================================================
# domain
 
L = 3000.0 # length, domain size
Lcenter = 0.0 # x value @ the center of the grid
N = 2^8 # series length (must be at least even)
dx = L/Float64(N) # length, uniform longitudinal grid spacing
x = collect(0.5*dx:dx:dx*Float64(N))-(L/2.0-Lcenter) # length, centered uniform grid


# =============================================================================
# test signals

ifield = 1

if ifield == 1 # cos(kx), machine precision for DCT
A = 3.0; lambda = L/2.0; ks = 2.0*pi/lambda; # rad/length
u = A*cos(ks*x); dudxA = -A*ks*sin(ks*x); d2udx2A = -A*ks^2.0*cos(ks*x); 
elseif ifield == 2 # sin(kx), difficult for DCT
A = 3.0 # m/s, velocity signal amplitude
lambda = L/10.0 # km, wavelength of signal (smallest possible)
ks = 2.0*pi/lambda # rad/km
u = A*sin(ks*x) # km/s
dudxA = A*ks*cos(ks*x) 
d2udx2A = -A*ks^2.0*sin(ks*x)
elseif ifield == 3 # Gaussian(x)
sigma = L/20.0
u = exp(-(x-Lcenter).^2.0/(2.0*sigma^2.0)) # test signal equivalent to gaussmf(x,[L/10 0]);
dudxA = (x-Lcenter).*u.*(-sigma^(-2.0)) # test signal derivative (Gaussian)
d2udx2A = u.*((x-Lcenter).^2.0-sigma^2.0)./sigma^4.0
end
 
# signal plot 
fig = figure(); CP = plot(x./L,u,"k");
xlabel("x"); ylabel("u"); title("signal");
savefig("./figures/signal.png",format="png"); close(fig);


# =============================================================================
# sine/cosine transforms

Ucos = r2r(u,FFTW.REDFT10)./(2.0*Float64(N)); # DCT-II 
Usin = r2r(u,FFTW.RODFT10)./(2.0*Float64(N)); # DST-II 

# wavenumbers 
kc = collect(0:Int64(N)-1).*(pi/L); # for DCT-II
ks = collect(1:Int64(N)).*(pi/L); # for DST-II
hc = kc./pi; # 1/length
kcinv = copy(kc); kcinv[1] = Inf;

# comparison of transforms
fig = figure(); plot(kc,Ucos,"r",label="DCT-II"); 
plot(ks,Usin,"b",label="DST-II");
xlabel("k (rads/m)"); title("transforms"); legend();
savefig("./figures/spectral_dst_dct.png",format="png"); close(fig);


# =============================================================================
# inverse discrete cosine transform

uinv = r2r(Ucos,FFTW.REDFT01); # inverse dst

# comparison of time series and reconstructed time series 
fig = figure(); plot(x./L,u,"k",label="signal");
plot(x./L,uinv,"b",label="DCT => IDCT");
xlabel("x"); title("discrete cosine transform"); legend();
savefig("./figures/reconstructed_signal.png",format="png"); close(fig);

# DST => IDST error plot
fig = figure(); CP = semilogy(x./L,abs(u-real(uinv)),"k");
xlabel("x"); title("discrete cosine transform error");
savefig("./figures/reconstructed_signal_error.png",format="png"); close(fig);


# =============================================================================
# first derivative

Ushift = [Ucos[2:Int64(N)];0.0]; # cosine shift (pi/2) for 1st derivative

# plot of transformed derivative
fig = figure(); plot(kc,Ucos,"k",label="cosine transformed signal");
plot(ks,-Ushift,"b",label="transformed derivative");
xlabel("k (rads/length)"); title("du/dx transform"); legend();
savefig("./figures/spectral_first_derivative.png",format="png"); close(fig);

# signal derivative via cosine transform 
#dudx = r2r(-Ushift.*ks,FFTW.RODFT01) # DST-III (inverse dst) 
dudx = dct_d1(Ucos,ks);

# comparison of analytical du/dx and computed du/dx plot
fig = figure(); plot(x./L,dudxA,"k",label="signal");
plot(x./L,dudx,"b--",label="IDST"); xlabel("x"); 
title("du/dx by DST-III (IDST) of shifted DCT-II"); legend();
savefig("./figures/first_derivative.png",format="png"); close(fig);

deriv_error = abs(dudxA-dudx); max_deriv_error = maximum(abs(dudxA-dudx));
println("\nThe maximum first derivative error is $(max_deriv_error) for $N grid points\n");

# first derivative error plot
fig = figure(); CP = semilogy(x./L,deriv_error,"k");
xlabel("x"); title("du/dx by DST-III (IDST) of shifted DCT-II, error");
savefig("./figures/first_derivative_error.png",format="png"); close(fig);


# =============================================================================
# second derivative

d2udx2 = dct_d2(Ucos,kc);

# plot of transformed derivative
fig = figure(); plot(kc,Ucos,"k",label="cosine transformed signal");
plot(kc,-Ucos.*kc.^2.0,"b",label="transformed derivative"); 
xlabel("k (rads/length)"); title("d^2u/dx^2 transform"); legend();
savefig("./figures/spectral_second_derivative.png",format="png"); close(fig);

# comparison with analytical d^2u/dx^2
fig = figure(); plot(x./L,d2udx2A,"k",label="signal");
plot(x./L,d2udx2,"b",label="IDCT"); 
xlabel("x"); title("d^2u/dx^2 by DCT-III (IDCT) of DCT-II"); legend();
savefig("./figures/second_derivative.png",format="png"); close(fig);

deriv2_error = abs(d2udx2A-d2udx2); max_deriv2_error = maximum(abs(d2udx2A-d2udx2));
println("The maximum second derivative error is $(max_deriv2_error) for $N grid points\n");

# error plot
fig = figure(); semilogy(x./L,abs(d2udx2A-d2udx2),"k");
xlabel("x"); title("d^2u/dx^2 by DCT-III (IDCT) of DCT-II, error");
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

Ua = r2r(ua,FFTW.REDFT10)./(2.0*Float64(N)); # DCT-II 
Ub = r2r(ub,FFTW.REDFT10)./(2.0*Float64(N)); # DCT-II 

# random signal plot 
fig = figure(); plot(x./L,ua,"r",label="u_1");
plot(x./L,ub,"b",label="u_2");
xlabel("x (km)"); legend(); title("random signals");
savefig("./figures/random_signals.png",format="png"); close(fig);

# single-sided spectrum plot
fig = figure(); semilogx(hc,2.0/Float64(N)*abs(Ua),"r",label="u_1");
semilogx(hc,2.0/Float64(N)*abs(Ub),"b",label="u_2"); ylabel("|U|");
xlabel("spatial frequency (1/length)"); legend(loc=2); title("Single sided spectrum");
savefig("./figures/single_sided_white_spectrums.png",format="png"); close(fig);

u2_alias = ua.*ub; # aliased square
u2_dealias = dealias(Ua,Ub,kc); # de-aliased square
U2_alias = r2r(u2_alias,FFTW.REDFT10)./(2.0*Float64(N)); # DCT-II 
U2_dealias = r2r(u2_dealias,FFTW.REDFT10)./(2.0*Float64(N)); # DCT-II 

# alias vs. de-aliased physical plot 
fig = figure(); plot(x./L,u2_alias,"r",label="aliased");
plot(x./L,u2_dealias,"b",label="de-aliased");
xlabel("x"); legend(); title("u_1*u_2");
savefig("./figures/aliased_signals.png",format="png"); close(fig);

# single-sided spectrum plot
fig = figure(); semilogx(hc,2.0/Float64(N)*abs(U2_alias),"r",label="aliased");
semilogx(hc,2.0/Float64(N)*abs(U2_dealias),"b",label="de-aliased"); ylabel("|U^2|");
xlabel("spatial frequency (1/length)"); legend(loc=2); title("u_1*u_2");
#axis([pi/(4.0*Lx),float(N)*pi/(5.0*Lx),0.0,0.05]);
savefig("./figures/aliasing_spectrums.png",format="png"); close(fig);


# =============================================================================
# Poisson equation solution by dct

# Poisson equation solution: Laplacian(psi) = qA
uP = poisson(d2udx2A,kcinv); 
Poisson_error = abs(uP-u); # Poisson equation solution error
max_err = maximum(Poisson_error);

fig = figure(); plot(x./L,u,"k",label="signal"); 
plot(x./L,uP,"b",label="Poisson solution"); legend();
xlabel("x"); ylabel("u"); title("Laplacian(psi) = q, psi solution"); 
savefig("./figures/Poisson_solution.png",format="png"); close(fig);

fig = figure(); CP = semilogy(x./L,Poisson_error,"k")
xlabel("x"); ylabel("error"); title("Laplacian(psi) = q, psi solution error"); 
savefig("./figures/Poisson_solution_error.png",format="png"); close(fig);

println("The maximum Poisson equation computation error is $(max_err) for $N gridpoints.\n")


