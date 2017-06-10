# Derivatives, spectral anti-derivatives, and de-aliasing by discrete Fourier
# transforms in 2D. 
# Bryan Kaiser 
# 3/25/17

using PyCall
using PyPlot
#@pyimport numpy as np
#@pyimport pylab as py


# =============================================================================
# readme:

# This script shows how to take 2D derivatives, do 2D spectral inversions, 
# and how to de-alias nonlinear signals using FFTW. At the end there is an
# additional Poisson equation solution for Gaussian on a linear slope 
# (e.g. a beta plane in GFD)

# Make sure that you create the directory "/figures" in the same directory as 
# as this script, for output plots.


# =============================================================================
# choice of test signals

# test signal for derivative and inversion examples:
ifield = 2; # enter 1) for a 2D Gaussian signal or 2) for a 2D sine wave

# test signal for de-aliasing example:
nonlinear_signal = 1; # enter 1) for 2D sine waves with noise or 2) for 2D 
# Gaussians with noise.


# =============================================================================

function meshgrid{T}(vx::AbstractVector{T}, vy::AbstractVector{T})
	# the same as the MATLAB function
    	m, n = length(vy), length(vx)
    	vx = reshape(vx, 1, n)
    	vy = reshape(vy, m, 1)
    	(repmat(vx, m, 1), repmat(vy, 1, n))
end

function power_spectrum_2D(S::Array{Float64,2},Kmag::Array{Float64,2})
	# takes 2D spectra and generates a 1D power spectra for plotting. 
	# convert the gridded wavenumber magnitudes to a vector, remove 
	# repeated values, and sort:
	Kmag_vec_union = sort(union(vec(Kmag))); 
	S_vec = zeros(size(Kmag_vec_union)); # power spectrum
	for j = 1:length(Kmag_vec_union) # each wavenumber magnitude
		s = 0.0; count = 0.0;
		for n = 1:(size(Kmag,1)*size(Kmag,2)) # loop over Kmag, S
			if Kmag[n] == Kmag_vec_union[j];
				s = s + S[n];
				count = count+1.0;
			end
		end 
		S_vec[j] = s/count; # averaged magnitude
	end
	return S_vec, Kmag_vec_union;
end

function dealias(U::Array{Complex{Float64},2},V::Array{Complex{Float64},2},mask::Array{Complex{Float64},2})
	uv = real(ifft(U.*mask,1:2)).*real(ifft(V.*mask,1:2));
	return uv;
end

function poisson(q::Array{Float64,2},K::Array{Float64,2},L::Array{Float64,2})
	# Uses the 2D fft to solve Laplacian(psi) = q for psi.
	#psi = real(ifft(-fft(q,1:2).*((K.^2.0+L.^2.0).^(-1.0))));
	psi = real(ifft(-fft(fft(q,1),2).*((K.^2.0+L.^2.0).^(-1.0))));
	return psi
end


# =============================================================================
# domain 

Lx = 3000.0; Ly = Lx; # km, domain size
Lxcenter = 0.0; Lycenter = 0.0; # x & y values at the center of the grid
Nx = 2^5; # series length (must be at least even)
Ny = 2^5; # series length (must be at least even)
dx = Lx/Float64(Nx); dy = Ly/Float64(Ny); # m, grid spacing
x = collect(0.5*dx:dx:dx*Float64(Nx))-(Lx/2.0-Lxcenter); # m, 
y = collect(0.5*dy:dy:dy*Float64(Ny))-(Ly/2.0-Lxcenter); # m, 
X,Y = meshgrid(x,y); 


# =============================================================================
# choice of test signal for derivatives and spectral inversions

if ifield == 1 # A 2D Gaussian test case ("A" for analytical solutions)
sigma = Lx/20.0; psi = exp(-((X-Lxcenter).^2.0+(Y-Lycenter).^2.0)./(2.0*sigma^2.0));
div_psiA = (X-Lxcenter+Y-Lycenter).*psi.*(-sigma^(-2.0)); 
dpsidxA = -(X-Lxcenter).*psi.*(sigma^(-2.0)); 
dpsidyA = -(Y-Lycenter).*psi.*(sigma^(-2.0)); 
qA = psi.*(((X-Lxcenter).^2.0+(Y-Lycenter).^2.0).*sigma^(-4.0)-2.0*sigma^(-2.0)); 
elseif ifield == 2 # a 2D sine wave
kx = (2.0*pi/Lx); ky = (2.0*pi/Ly); u = sin(X.*kx).*sin(Y.*ky);
divergenceA = (sin(X.*kx).*cos(Y.*ky).*ky+cos(X.*kx).*sin(Y.*ky).*kx);
dudxA =  kx.*cos(X.*kx).*sin(Y.*ky); dudyA =  ky.*sin(X.*kx).*cos(Y.*ky);
laplacianA = -u.*(kx^(2.0)+ky^(2.0));
end 


# =============================================================================
# plots of the signal:

fig = figure(); CP = contourf(X,Y,u,200,cmap="RdBu"); 
xlabel("x (km)"); ylabel("y (km)"); title("u, signal"); colorbar(CP); 
savefig("./figures/signal.png",format="png"); close(fig);

# u=-dpsi/dy, analytical
fig = figure(); CP = contourf(X./Lx,Y./Ly,dudyA,200,cmap="RdBu"); 
xlabel("x"); ylabel("y"); title("signal y derivative"); colorbar(CP); 
savefig("./figures/y_derivative_signal.png",format="png"); close(fig);

# v=dpsi/dx, analytical 
fig = figure(); CP = contourf(X./Lx,Y./Ly,dudxA,200,cmap="RdBu"); 
xlabel("x"); ylabel("y"); title("signal x derivative"); colorbar(CP); 
savefig("./figures/x_derivative_signal.png",format="png"); close(fig);

# divergence of psi analytical 
fig = figure(); CP = contourf(X./Lx,Y./Ly,divergenceA,200,cmap="RdBu"); 
xlabel("x"); ylabel("y"); title("divergence of u analytical");
colorbar(CP); savefig("./figures/divergence_signal.png",format="png"); close(fig);

# Laplacian of psi q=nabla2(psi) analytical 
fig = figure(); CP = contourf(X./Lx,Y./Ly,laplacianA,200,cmap="PuOr"); 
xlabel("x"); ylabel("y"); title("Laplacian of signal"); 
colorbar(CP); savefig("./figures/Laplacian_signal.png",format="png");


# =============================================================================
# 2D FFT of psi(x,y) 

# Fourier transform of psi into PSI 
U = fft(u,1:2); # 2D fft
Ux = fft(u,2); Uy = fft(u,1); # 1D fft

# wavenumbers 
k = zeros(Nx); l = zeros(Ny);
k[2:Int32(Nx/2)+1] = collect(1:Int32(Nx/2)).*(2.0*pi/Lx); # rad/m
k[Int32(Nx/2)+2:Nx] = -collect(Int32(Nx/2)-1:-1:1).*(2.0*pi/Lx); # rad/m
l[2:Int32(Ny/2)+1] = collect(1:Int32(Ny/2)).*(2.0*pi/Ly); # rad/m
l[Int32(Ny/2)+2:Ny] = -collect(Int32(Ny/2)-1:-1:1).*(2.0*pi/Ly); # rad/m
K,L = meshgrid(k,l); 
Kmag = (K.^2.0+L.^2.0).^(1.0/2.0); # wavenumber magnitude
Kmag_shift = fftshift(Kmag);
Hk = K./(2.0*pi); Hl = L./(2.0*pi); Hmag = Kmag./(2.0*pi);
Kinv = copy(K); Kinv[1,1] = Inf; Linv = copy(L); Linv[1,1] = Inf;

fig = figure(); CP = contourf(1:Nx,1:Ny,Kmag,100,cmap="spectral"); 
xlabel("1:N"); ylabel("1:N"); title("|K|"); colorbar(CP); 
savefig("./figures/wavenumber_magnitude.png",format="png"); close(fig);

 
# =============================================================================
# Fourier domain and power spectrum plots

P=(abs(U).*(2.0/Float64(Nx*Ny))).^2.0; 
P_vec, Kmag_vec_un = power_spectrum_2D(P,Kmag);

fig = figure(); CP = plot(Kmag_vec_un,P_vec,"b");
xlabel("|K|"); ylabel("|U|"); title("2D power spectrum"); 
savefig("./figures/power_spectrum.png",format="png"); close(fig);

fig = figure(); CP = contourf(K,L,P,100,cmap="gray");
xlabel("k (2pi/L) rad/km"); ylabel("l (2pi/L) rad/km"); 
title("|U| (unshifted, no padding)"); colorbar(CP); 
savefig("./figures/power_spectrum_2D.png",format="png"); close(fig);


# =============================================================================
# divergence by fft of psi(x,y)

# first derivatives by fft
dxu = real(ifft(Ux.*K.*im,2)); dyu = real(ifft(Uy.*L.*im,1));
#divergence = real(ifft(U.*(K+L).*im,1:2)); 
divergence = dxu+dyu;

# divergence error
divergence_error = abs(divergenceA-divergence); max_div_error = maximum(divergence_error);
println("The maximum divergence computation error is $(max_div_error) for a $Nx by $Ny grid.\n")

# first derivative plots:

#=
# u=-dpsi/dy, error 
fig = figure(); CP = contourf(X./Lx,Y./Ly,abs(dypsi-dpsidyA),200,cmap="gray"); 
xlabel("x"); ylabel("y"); title("y derivative, error"); colorbar(CP); 
savefig("./figures/y_derivative_error.png",format="png"); close(fig);

# v=dpsi/dx, error
fig = figure(); CP = contourf(X./Lx,Y./Ly,abs(dxpsi-dpsidxA),200,cmap="gray"); 
xlabel("x"); ylabel("y"); title("x derivative, error"); colorbar(CP); 
savefig("./figures/x_derivative_error.png",format="png"); close(fig);
=#

# divergence of psi error
fig = figure(); CP = contourf(X./Lx,Y./Ly,divergence_error,200,cmap="gray");
xlabel("x"); ylabel("y"); title("divergence, error");
colorbar(CP); savefig("./figures/divergence_error.png",format="png"); close(fig);


# =============================================================================
# Laplacian by fft of psi(x,y)

# the Laplacian by inverse fft
laplacian = real(ifft(-Ux.*K.^2.0,2)+ifft(-Uy.*L.^2.0,1));
#laplacian = real(ifft(-U.*(K.^2.0+L.^2.0),1:2));

# maximum error
laplacian_error = abs(laplacianA-laplacian); max_lap_error = maximum(laplacian_error);
println("The maximum Laplacian computation error is $(max_lap_error) for a $Nx by $Ny grid.\n")

# plot of the real component, computational error 
fig = figure(); CP = contourf(X./Lx,Y./Ly,laplacian_error,200,cmap="gray");
xlabel("x"); ylabel("y"); title("Laplacian error"); 
colorbar(CP); savefig("./figures/Laplacian_error.png",format="png");


# =============================================================================
# De-aliasing a nonlinear (quadratic) signal

if nonlinear_signal == 1 # sine waves with random noise
kx = (2.0*pi/Lx); ky = (2.0*pi/Ly); 
ua = rand(size(X)).*0.5+sin(X.*kx).*sin(Y.*ky);
ub = rand(size(X)).*0.5+sin(X.*kx).*sin(Y.*ky);
elseif nonlinear_signal == 2 # Gaussian with random noise
sigma = Lx/10.0; 
ua = exp(-((X-Lxcenter).^2+(Y-Lycenter).^2)./(2.0*sigma^2))+rand(size(X)).*0.5;
ub = exp(-((X-Lxcenter).^2+(Y-Lycenter).^2)./(2.0*sigma^2))+rand(size(X)).*0.5;
end

# mask for 2/3 rule padding for de-aliasing a quadratic signal via fft
mask = ones(size(Kmag))+ones(size(Kmag)).*im;
for j = 1:(size(Kmag,1)*size(Kmag,2)) # 2/3 rule
	if abs(Kmag[j]) >= max(size(Kmag,1),size(Kmag,2))/3.0*Kmag[1,2]; 
		mask[j] = 0.0+0.0im; # Fourier space variables
	end
end

# aliased and de-aliased quadratic signal
u2_alias = ua.*ub; # aliased square
Ua = fft(ua,1:2); Ub = fft(ub,1:2);
tic(); u2_dealias = dealias(Ua,Ub,mask); time = toq(); # de-aliased square
println("The computation time is for the de-aliased signal is $time seconds for a $Nx by $Ny grid.\n")

# fft of the quadratic signal
S_alias = fft(u2_alias,1:2); 
S1 = (abs(S_alias)).*2.0/Float64((Nx^2.0+Ny^2.0)^(1.0/2.0)); 
S_dealias = fft(u2_dealias,1:2); 
S2 = (abs(S_dealias)).*2.0/Float64((Nx^2.0+Ny^2.0)^(1.0/2.0)); 

# 1D power spectrum of 2D quadratic signals
S1_vec, Hmag_vec1 = power_spectrum_2D(S1,Hmag);
S2_vec, Hmag_vec2 = power_spectrum_2D(S2,Hmag);

# plots:

fig = figure(); semilogx(Hmag_vec1,S1_vec,"r",label="aliased");
semilogx(Hmag_vec2,S2_vec,"b",label="de-aliased"); legend();
xlabel("k"); ylabel("|PSI|"); title("2D power spectrum"); 
axis([minimum(Hmag_vec1),maximum(Hmag_vec1),0.0,maximum(S1_vec)/2.0]);
savefig("./figures/quadratic_signal_power_spectrum.png",format="png"); close(fig);

fig = figure(); CP = contourf(X./Lx,Y./Ly,u2_alias,200,cmap="PuOr"); 
xlabel("x"); ylabel("y"); title("u*u aliased"); 
colorbar(CP); savefig("./figures/quadratic_signal_aliased.png",format="png"); 

fig = figure(); CP = contourf(X./Lx,Y./Ly,u2_dealias,200,cmap="PuOr"); 
xlabel("x"); ylabel("y"); title("u*u de-aliased"); 
colorbar(CP); savefig("./figures/quadratic_signal_dealiased.png",format="png");


# =============================================================================
# Poisson equation solution by fft

# Poisson equation solution: Laplacian(psi) = qA
tic(); uP = poisson(laplacianA,Kinv,Linv); time = toq(); 
println("The first Poisson equation computation time is $(time) seconds for a $Nx by $Ny grid.\n")
Poisson_error = abs(uP-u); # Poisson equation solution error
max_err = maximum(Poisson_error);

fig = figure(); CP = contourf(X./Lx,Y./Ly,uP,200,cmap="Spectral");
xlabel("x"); ylabel("y"); title("Laplacian(psi) = q, psi solution"); 
colorbar(CP); savefig("./figures/Poisson_solution.png",format="png"); 

fig = figure(); CP = contourf(X./Lx,Y./Ly,Poisson_error,200,cmap="gray")
xlabel("x"); ylabel("y"); title("Laplacian(psi) = q, psi solution error"); 
colorbar(CP); savefig("./figures/Poisson_solution_error.png",format="png"); 

println("The maximum Poisson equation computation error is $(max_err) for a $Nx by $Ny grid.\n")

# Another example: a Gaussian on a linear y slope (beta plane):
sigma = Lx/20.0; beta = 1E-9;
psi2 = exp(-((X-Lxcenter).^2.0+(Y-Lycenter).^2.0)./(2.0*sigma^2.0));
qA2 = psi2.*(((X-Lxcenter).^2.0+(Y-Lycenter).^2.0).*sigma^(-4.0)-2.0*sigma^(-2.0))-Y.*beta; 

# Poisson equation solution: Laplacian(psi) = qA
tic(); psiP2 = poisson(qA2,Kinv,Linv); time = toq(); 
println("The second Poisson equation computation time is $time seconds for a $Nx by $Ny grid.\n")

fig = figure(); CP = contourf(X./Lx,Y./Ly,psiP2,200,cmap="RdBu");
xlabel("x"); ylabel("y"); title("Laplacian(psi) = q-By, psi solution"); 
colorbar(CP); savefig("./figures/Poisson_solution_linear_slope.png",format="png");


# =============================================================================
# fft plan speed test

m_end = 1000;
nx = collect(4:10); # typeof(n) = Array{Int64,1}, powers of 2 grid resolution
ny = collect(4:10);

# initialized fields
fft_time = zeros(length(nx)); fft_plan_time = zeros(length(nx));
Linf_div = zeros(length(nx)); Linf_divP = zeros(length(nx));
Linf_lap = zeros(length(nx)); Linf_lapP = zeros(length(nx));

for m = 1:length(nx) # loop over powers of 2 grid resolution 

# domain
Lx = 3000.0; Ly = Lx; # km, domain size
Lxcenter = 0.0; Lycenter = 0.0; # x & y values at the center of the grid
Nx = 2^(nx[m]); # series length (must be at least even)
Ny = 2^(ny[m]); # series length (must be at least even)
dx = Lx/Float64(Nx); dy = Ly/Float64(Ny); # m, grid spacing
x = collect(0.5*dx:dx:dx*Float64(Nx))-(Lx/2.0-Lxcenter); # m, 
y = collect(0.5*dy:dy:dy*Float64(Ny))-(Ly/2.0-Lxcenter); # m, 
X,Y = meshgrid(x,y); 

# signal
kx = (2.0*pi/Lx); ky = (2.0*pi/Ly); u = sin(X.*kx).*sin(Y.*ky);
divergenceA = (sin(X.*kx).*cos(Y.*ky).*ky+cos(X.*kx).*sin(Y.*ky).*kx);
dudxA =  kx.*cos(X.*kx).*sin(Y.*ky); dudyA =  ky.*sin(X.*kx).*cos(Y.*ky);
laplacianA = -u.*(kx^(2.0)+ky^(2.0));

# wavenumbers 
k = zeros(Nx); l = zeros(Ny);
k[2:Int64(Nx/2)+1] = collect(1:Int64(Nx/2)).*(2.0*pi/Lx); # rad/m
k[Int64(Nx/2)+2:Nx] = -collect(Int64(Nx/2)-1:-1:1).*(2.0*pi/Lx); # rad/m
l[2:Int64(Ny/2)+1] = collect(1:Int64(Ny/2)).*(2.0*pi/Ly); # rad/m
l[Int64(Ny/2)+2:Ny] = -collect(Int64(Ny/2)-1:-1:1).*(2.0*pi/Ly); # rad/m
K,L = meshgrid(k,l); 
Kmag = (K.^2.0+L.^2.0).^(1.0/2.0); # wavenumber magnitude

# initialize spectral u, derivatives
U = zeros(size(X)); dudx = zeros(size(X)); d2udx2 = zeros(size(X));
Up = zeros(size(X)); dudxp = zeros(size(X)); d2udx2p = zeros(size(X));

# spectral u, derivatives via standard fft: wall time test
tic(); for m1 = 1:m_end;  
Ux = fft(u,2); Uy = fft(u,1); # 1D fft
divergence = real(ifft(Ux.*K.*im,2))+real(ifft(Uy.*L.*im,1));
laplacian = real(ifft(-Ux.*K.^2.0,2)+ifft(-Uy.*L.^2.0,1));
end; fft_time[m] = toq();

# multithread number of threads and fft_plan:
FFTW.set_num_threads(4)
PUx = plan_fft(u,2,flags=FFTW.EXHAUSTIVE); # 1D fft
PUy = plan_fft(u,1,flags=FFTW.EXHAUSTIVE); # 1D fft 
UxP = PUx*u; Uyp = PUy*u; 
Pdx = plan_ifft(Ux.*K.*im,2,flags=FFTW.EXHAUSTIVE);
Pdy = plan_ifft(Uy.*L.*im,1,flags=FFTW.EXHAUSTIVE);
divergenceP = real(Pdx*(Ux.*K.*im)+Pdy*(Uy.*L.*im)); 
Pdx2 = plan_ifft(-Ux.*K.^2.0,2,flags=FFTW.EXHAUSTIVE);
Pdy2 = plan_ifft(-Uy.*L.^2.0,1,flags=FFTW.EXHAUSTIVE);
laplacianP = real(Pdx2*(-Ux.*K.^2.0)+Pdy2*(-Uy.*L.^2.0)); 

# spectral u, derivatives via fft plan: wall time test
tic(); for m2 = 1:m_end; 
UxP = PUx*u; Uyp = PUy*u; 
divergenceP = real(Pdx*(Ux.*K.*im)+Pdy*(Uy.*L.*im));
laplacianP = real(Pdx2*(-Ux.*K.^2.0)+Pdy2*(-Uy.*L.^2.0)); 
end; fft_plan_time[m] = toq();

# infinity norm error for fft and fft plan 
Linf_div[m] = maximum(abs(divergence-divergenceA));
Linf_lap[m] = maximum(abs(laplacian-laplacianA));
Linf_divP[m] = maximum(abs(divergenceP-divergenceA));
Linf_lapP[m] = maximum(abs(laplacianP-laplacianA));

end

Nvec = (ones(length(nx)).*2.0).^nx;
NlogN = (log2(Nvec).*Nvec).*((fft_time[1]/(log2(Nvec[1]).*Nvec[1]))*10.0);

# wall time plot
fig = figure(); 
loglog(Nvec.^2.0,NlogN,"k--",label="Nlog2(N)");
loglog(Nvec.^2.0,fft_time,"r",label="2D fft"); 
loglog(Nvec.^2.0,fft_plan_time,"b",label="2D fft plan"); legend(loc=2);
xlabel("N^2"); ylabel("wall time, seconds"); title("$m_end divergence/Laplacian calculations"); 
savefig("./figures/fft_plan_computation_time.png",format="png"); close(fig);

# error plot
fig = figure(); 
loglog(Nvec.^2.0,Linf_div,"k",label="divergence"); 
loglog(Nvec.^2.0,Linf_lap,"g",label="Laplacian"); 
loglog(Nvec.^2.0,Linf_divP,"b--",label="divergence, plan");
loglog(Nvec.^2.0,Linf_lapP,"r--",label="Laplacian, plan"); 
xlabel("N^2"); ylabel("L infinity"); title("2D fft error"); legend(loc=4);
savefig("./figures/fft_plan_error.png",format="png"); close(fig)

println("done!")

