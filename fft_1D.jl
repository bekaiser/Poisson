# This script shows how to take 1D derivatives by fft as well as how to solve
# the 1D Poisson equation, and how to multi-thread the fft operations.
# Bryan Kaiser 
# 3/10/17

using DataArrays
using PyPlot
using PyCall
using Base.FFTW 


# =============================================================================
# functions

function dealias(U,V,mask)
	uv = real(ifft(U.*mask)).*real(ifft(V.*mask)); # de-aliased product
	return uv
end

function poisson(q::Array{Float64,1},k::Array{Float64,1})
	# Uses the 2D fft to solve Laplacian(psi) = q for psi.
	psi = real(ifft(-fft(q).*(k.^(-2.0))));
	return psi
end

# =============================================================================
# domain

L = 3000.0; # km, domain size
Lcenter = 0.0; # x value @ the center of the grid
N = 2^10; # series length (must be at least even)
dx = L/Float64(N); # km, uniform longitudinal grid spacing
x = collect(0.5*dx:dx:dx*N)-(L/2.0-Lcenter); # km, centered uniform grid 
 

# =============================================================================
# test signals

ifield = 2
# 1 for cos(kx), 2 for sin(kx), 3 for Gaussian

if ifield == 1 # cos(kx) 
A = 3.0 # m/s, velocity signal amplitude
lambda = L/10.0 # km, wavelength of signal (smallest possible)
ks = 2.0*pi/lambda # rad/km
u = A*cos(ks*x) # km/s (not physical...)
du = -A*ks*sin(ks*x) 
d2u = -A*ks^2.0*cos(ks*x) 

elseif ifield == 2 # sin(kx)
A = 3.0 # m/s, velocity signal amplitude
lambda = L/10.0 # km, wavelength of signal (smallest possible)
ks = 2.0*pi/lambda # rad/km
u = A*sin(ks*x) # km/s
du = A*ks*cos(ks*x) 
d2u = -A*ks^2.0*sin(ks*x)

elseif ifield == 3 # Gaussian
sigma = L/20.0
u = exp(-x.^2.0/(2.0*sigma^2.0)) # test signal equivalent to gaussmf(x,[L/10 0]);
du = x.*u.*(-sigma^(-2.0)) # test signal derivative (Gaussian)
d2u = u.*(x.^2.0-sigma^2.0)./sigma^4.0;

end # ifield choice

# signal plot 
fig = figure(); CP = plot(x./L,u,"k");
xlabel("x (km)"); ylabel("u"); title("signal");
savefig("./figures/signal.png",format="png"); close(fig);


# =============================================================================
# Fourier transform

U = fft(u);
#FFTW.set_num_threads(4) # multi-threading for higher performance

# wavenumbers
k = zeros(N);
k[2:Int32(N/2)+1] = collect(1:Int32(N/2)).*(2.0*pi/L); # rad/km
k[Int32(N/2)+2:N] = -collect(Int32(N/2)-1:-1:1).*(2.0*pi/L); # rad/km

hz = k./(2.0*pi) # 1/length, equivalent to Hz for time

# full spectrum plot
fig = figure(); CP = plot(hz, 2.0/float(N).*abs(U),"k");
xlabel("spatial frequency (1/length)"); ylabel("|U|"); title("Full spectrum");
savefig("./figures/full_spectrum.png",format="png"); close(fig);

# single-sided spectrum plot
fig = figure(); CP = semilogx(hz,2.0/float(N)*abs(U),"k")
xlabel("spatial frequency (1/length)"); ylabel("|U|"); title("Single sided spectrum");
savefig("./figures/single_sided_spectrum.png",format="png"); close(fig);


# =============================================================================
# inverse Fourier transform

uinv = real(ifft(U))

# comparison of time series and reconstructed time series plot
fig = figure(); plot(x./L,u,"k",label="signal"); plot(x./L,uinv,"b",label="ifft");
xlabel("x"); title("inverse Fourier transform"); legend();
savefig("./figures/inverse_transform.png",format="png"); close(fig);

# error plot
fig = figure(); CP = semilogy(x./L,abs(u-uinv),"k");
xlabel("x"); title("inverse Fourier transform error")
savefig("./figures/inverse_transform_error.png",format="png"); close(fig);


# =============================================================================
# first derivative

# signal derivative via Fourier transform
dudx = real(ifft(U.*k.*im));

# comparison of analytical du/dx and computed du/dx plot
fig = figure(); plot(x./L,du,"b",label="signal");
plot(x./L,dudx,"r--",label="computed"); legend();
xlabel("x"); title("first derivative by fft"); axis([-0.1,0.1,-0.08,0.08]);
savefig("./figures/first_derivative.png",format="png"); close(fig);

# error plot
fig = figure(); CP = semilogy(x./L,abs(du-real(dudx)),"k");
xlabel("x"); title("first derivative error");
savefig("./figures/first_derivative_error.png",format="png"); close(fig);


# =============================================================================
# second derivative

d2udx2 = real(ifft(-U.*k.^2.0));

# comparison with analytical d^2u/dx^2
fig = figure(); plot(x./L,d2u,"b",label="signal");
plot(x./L,d2udx2,"r--",label="computed"); legend();
xlabel("x"); title("second derivative by fft"); axis([-0.1,0.1,-0.002,0.002]);
savefig("./figures/second_derivative.png",format="png"); close(fig);

# error plot
fig = figure(); CP = semilogy(x./L,abs(d2u-real(d2udx2)),"k");
xlabel("x"); title("second derivative error");
savefig("./figures/second_derivative_error.png",format="png"); close(fig);

readline()
# =============================================================================
# nonlinearity and de-aliasing

alias_signal = 1;

if alias_signal == 1 # random signals
A = 1.0; # m/s, velocity signal amplitude
ua = rand(size(x)).*A; ub = rand(size(x)).*A;
elseif alias_signal == 2 # harmonic signals
ka = float(N)/100.0*(2.0*pi/Lx);
ua = sin(x.*ka); ub = ua;
end

Ua = fft(ua); Ub = fft(ub);

# random signal plot 
fig = figure(); plot(x./L,ua,"r",label="u_a");
plot(x./L,ub,"b",label="u_b"); axis([-0.1,0.1,-0.2,1.2]);
xlabel("x"); legend(); title("random signals");
savefig("./figures/random_signals.png",format="png"); close(fig);

# single-sided spectrum plot
fig = figure(); semilogx(hz,2.0/Float64(N)*abs(Ua),"r",label="u_a");
semilogx(hz,2.0/Float64(N)*abs(Ub),"b",label="u_b"); ylabel("|U|");
xlabel("spatial frequency (1/length)"); legend(loc=2); title("Single sided spectrum");
savefig("./figures/single_sided_white_spectrums.png",format="png"); close(fig);

# 2/3 zero mask
mask = ones(length(k))+ones(length(k)).*im;
for j = 1:length(k)
	if abs(k[j]) >= Float64(N)/3.0*k[2]; # 2/3 rule
		mask[j] = 0.0+0.0im;
	end
end

u2_alias = ua.*ub; # aliased square
u2_dealias = dealias(Ua,Ub,mask); # de-aliased square
U2_alias = fft(u2_alias); U2_dealias = fft(u2_dealias);

# alias vs. de-aliased physical plot 
fig = figure(); plot(x./L,u2_alias,"r",label="aliased");
plot(x./L,u2_dealias,"b",label="de-aliased");
xlabel("x"); legend(); title("u_a*u_b"); axis([-0.1,0.1,-0.2,1.2]);
savefig("./figures/aliased_signals.png",format="png"); close(fig);

# single-sided spectrum plot
fig = figure(); semilogx(hz,2.0/Float64(N)*abs(U2_alias),"r",label="aliased");
semilogx(hz,2.0/Float64(N)*abs(U2_dealias),"b",label="de-aliased"); ylabel("|U^2|");
xlabel("spatial frequency (1/length)"); legend(loc=2); title("u_a*u_b");
axis([2E-4,0.3,0.0,0.06]);
savefig("./figures/aliasing_spectrums.png",format="png"); close(fig);


# =============================================================================
# Poisson equation solution

# Poisson equation solution: Laplacian(psi) = qA
kinv = copy(k); kinv[1] = Inf;
uP = poisson(d2u,kinv);
Poisson_error = abs(uP-u); # Poisson equation solution error

max_err = maximum(Poisson_error);

fig = figure(); plot(x./L,u,"k",label="signal"); 
plot(x./L,uP,"b--",label="Poisson solution"); legend();
xlabel("x"); ylabel("u"); title("Laplacian(psi) = q, psi solution"); 
savefig("./figures/Poisson_solution.png",format="png"); close(fig);

fig = figure(); CP = plot(x./L,Poisson_error,"k")
xlabel("x"); ylabel("error"); title("Laplacian(psi) = q, psi solution error"); 
savefig("./figures/Poisson_solution_error.png",format="png"); close(fig);

println("The maximum Poisson equation computation error is $(max_err) for $N gridpoints.\n")


# =============================================================================
# multi-threading example

n = collect(3:18); # typeof(n) = Array{Int64,1}, powers of 2 grid resolution

# initialized fields
fft_time = zeros(length(n)); fft_plan_time = zeros(length(n));
Linf_d1 = zeros(length(n)); Linf_d2 = zeros(length(n));
Linf_d1P = zeros(length(n)); Linf_d2P = zeros(length(n));

for m = 1:length(n) # loop over powers of 2 grid resolution 

# grid
L = 3000.0; # km, domain size
Lcenter = 0.0; # x value @ the center of the grid
N = 2^n[m]; # series length (must be at least even)
dx = L/Float64(N); # km, uniform longitudinal grid spacing
x = collect(0.5*dx:dx:dx*N)-(L/2.0-Lcenter); # km, centered uniform grid 

# signal
if ifield == 1 # cos(kx) 
A = 3.0 # m/s, velocity signal amplitude
lambda = L/10.0 # km, wavelength of signal (smallest possible)
ks = 2.0*pi/lambda # rad/km
u = A*cos(ks*x) # km/s (not physical...)
du = -A*ks*sin(ks*x) 
d2u = -A*ks^2.0*cos(ks*x) 
elseif ifield == 2 # sin(kx)
A = 3.0 # m/s, velocity signal amplitude
lambda = L/10.0 # km, wavelength of signal (smallest possible)
ks = 2.0*pi/lambda # rad/km
u = A*sin(ks*x) # km/s
du = A*ks*cos(ks*x) 
d2u = -A*ks^2.0*sin(ks*x)
elseif ifield == 3 # Gaussian
sigma = L/20.0
u = exp(-x.^2.0/(2.0*sigma^2.0)) # test signal equivalent to gaussmf(x,[L/10 0]);
du = x.*u.*(-sigma^(-2.0)) # test signal derivative (Gaussian)
d2u = u.*(x.^2.0-sigma^2.0)./sigma^4.0;
end # ifield choice

# wavenumbers
k = zeros(N);
k[2:Int64(N/2)+1] = collect(1:Int64(N/2)).*(2.0*pi/L); # rad/km
k[Int64(N/2)+2:N] = -collect(Int64(N/2)-1:-1:1).*(2.0*pi/L); # rad/km

# initialize spectral u, derivatives
U = similar(x); dudx = similar(x); d2udx2 = similar(x);
Up = similar(x); dudxp = similar(x); d2udx2p = similar(x);

# spectral u, derivatives via standard fft: wall time test
tic(); for m1 = 1:100; U = fft(u); dudx = ifft(U.*k.*im); 
d2udx2 = ifft(-U.*k.^2.0);
end; fft_time[m] = toq();

# multithread number of threads and fft_plan:
FFTW.set_num_threads(4)
PU = plan_fft(u, flags=FFTW.MEASURE); Up = PU*u; 
Pdudx = plan_ifft(Up.*k.*im, flags=FFTW.MEASURE); 
Pd2udx2 = plan_ifft(-Up.*k.^2.0, flags=FFTW.MEASURE);

# spectral u, derivatives via fft plan: wall time test
tic(); for m2 = 1:100; Up = PU*u; dudxp = Pdudx*(Up.*k.*im); 
d2udx2p = Pd2udx2*(-Up.*k.^2.0);
end; fft_plan_time[m] = toq();

# infinity norm error for fft and fft plan 
Linf_d1[m] = maximum(abs(du-real(dudx)));
Linf_d2[m] = maximum(abs(d2u-real(d2udx2)));
Linf_d1P[m] = maximum(abs(du-real(dudxp)));
Linf_d2P[m] = maximum(abs(d2u-real(d2udx2p)));

end

# wall time plot
fig = figure(); plot(n,fft_time,"k",label="fft"); 
plot(n,fft_plan_time,"r--",label="plan_fft!"); legend();
xlabel("log_2(N)"); ylabel("seconds"); title("wall time, 1000 ffts"); 
savefig("./figures/fft_plan_computation_time.png",format="png"); close(fig);

# error plot
fig = figure(); semilogy(n,Linf_d1,"b--",label="dx, fft"); 
semilogy(n,Linf_d2,"b",label="dx2, fft"); 
semilogy(n,Linf_d1P,"r.",label="dx, fft plan"); 
semilogy(n,Linf_d2P,"r--",label="dx2, fft plan"); legend();
xlabel("log_2(N)"); ylabel("L infinity"); title("error"); 
savefig("./figures/fft_plan_error.png",format="png"); close(fig);


# =============================================================================
# add band averaging and confidence intervals

