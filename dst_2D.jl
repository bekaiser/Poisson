# Derivatives, spectral anti-derivatives, and de-aliasing by discrete sine 
# transforms in 2D. 
# Bryan Kaiser 
# 3/2/17

using PyPlot
using PyCall
using Base.FFTW
@pyimport numpy as np
@pyimport pylab as py


# =============================================================================
# readme:

# This script shows how to take 2D derivatives, do 2D spectral inversions, 
# and how to de-alias nonlinear signals using FFTW DSTs. At the end there is an
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
	m, n = length(vy), length(vx);
	vx = reshape(vx, 1, n); vy = reshape(vy, m, 1);
	(repmat(vx, m, 1), repmat(vy, 1, n));
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

function dealias(U::Array{Float64,2},V::Array{Float64,2},Kmags::Array{Float64,2})
	# 2/3 rule padding for de-aliasing a quadratic signal via dst
	for j = 1:(size(Kmags,1)*size(Kmags,2)) # 2/3 rule
		if abs(Kmags[j]) >= max(size(Kmags,1),size(Kmags,2))/3.0*Kmags[1,1]; 
			U[j] = 0.0; V[j] = 0.0; # Spectral domain variables
		end
	end
	return r2r(U,FFTW.RODFT01,1:2).*r2r(V,FFTW.RODFT01,1:2);
end

function poisson(q::Array{Float64,2},K::Array{Float64,2},L::Array{Float64,2})
	# 2D DST spectral inversion using FFTW to solve the Poisson equation:
	# given q and the Poisson equation, Laplacian(psi)=q, solve for psi.
	return r2r(-r2r(q,FFTW.RODFT10,1:2)./(4.0*Float64(size(q,1)*size(q,2))).*((K.^2+L.^2).^(-1.0)),FFTW.RODFT01,1:2); 
end



# =============================================================================
# domain
 
Lx = 3000.0; Ly = Lx; # m, domain size
Lxcenter = 0.0; Lycenter = 0.0; # x,y values @ the center of the grid
Nx = 2^7 # resolution, series length (must be at least even)
Ny = 2^7 # resolution, series length (must be at least even)
dx = Lx/Float64(Nx); dy = Ly/Float64(Ny);  # m, uniform grid spacings
x = collect(0.5*dx:dx:dx*Float64(Nx))-(Lx/2.0-Lxcenter); # m, centered uniform grid 
y = collect(0.5*dy:dy:dy*Float64(Ny))-(Ly/2.0-Lxcenter); # m, centered uniform grid
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
kx = (2.0*pi/Lx); ky = (2.0*pi/Ly); psi = sin(X.*kx).*sin(Y.*ky);
div_psiA = (sin(X.*kx).*cos(Y.*ky).*ky+cos(X.*kx).*sin(Y.*ky).*kx);
dpsidxA =  kx.*cos(X.*kx).*sin(Y.*ky); dpsidyA =  ky.*sin(X.*kx).*cos(Y.*ky);
qA = -psi.*(kx^(2.0)+ky^(2.0));
end 


# =============================================================================
# plots of the signal:

fig = figure(); CP = contourf(X./Lx,Y./Ly,psi,200,cmap="Spectral"); 
xlabel("x"); ylabel("y"); title("psi, signal"); colorbar(CP); 
savefig("./figures/signal.png",format="png"); close(fig);

fig = figure(); CP = contourf(X./Lx,Y./Ly,dpsidxA,200,cmap="RdBu");
xlabel("x"); ylabel("y"); title("signal x derivative"); 
colorbar(CP); savefig("./figures/x_derivative_signal.png",format="png"); 
close(fig);

fig = figure(); CP = contourf(X./Lx,Y./Ly,dpsidyA,200,cmap="RdBu");
xlabel("x"); ylabel("y"); title("signal y derivative"); 
colorbar(CP); savefig("./figures/y_derivative_signal.png",format="png"); 
close(fig);

fig = figure(); CP = contourf(X./Lx,Y./Ly,div_psiA,200,cmap="RdBu");
xlabel("x"); ylabel("y"); title("signal divergence"); 
colorbar(CP); savefig("./figures/divergence_signal.png",format="png"); 
close(fig);

fig = figure(); CP = contourf(X./Lx,Y./Ly,qA,200,cmap="PuOr");
xlabel("x"); ylabel("y"); title("Laplacian of signal"); 
colorbar(CP); savefig("./figures/Laplacian_signal.png",format="png"); 
close(fig);


# =============================================================================
# 2D discrete sine transform of psi(x,y) 

PSIsin = r2r(psi,FFTW.RODFT10,1:2)./(4.0*Float64(Nx*Ny)); # DST-II (2D)
PSIsinx = r2r(psi,FFTW.RODFT10,2)./(2.0*Float64(Nx)); # DST-II (1D)
PSIsiny = r2r(psi,FFTW.RODFT10,1)./(2.0*Float64(Ny)); # DST-II (1D)

psiinv = r2r(PSIsin,FFTW.RODFT01,1:2); # DST-III (IDST 2D)
idst_error = abs(psi-psiinv);

# inverse dst error
fig = figure(); CP = contourf(X./Lx,Y./Ly,idst_error,200,cmap="gray"); 
xlabel("x"); ylabel("y"); title("IDST error"); colorbar(CP); 
savefig("./figures/idst_error.png",format="png"); close(fig);


# =============================================================================
# wavenumbers for derivatives/inversion

# sine transform wavenumbers
ks = collect(1:Nx).*(pi/Lx); # for DST-II
ls = collect(1:Ny).*(pi/Ly); # for DST-II
Ks,Ls = meshgrid(ks,ls); 
Kmags = (Ks.^2.0+Ls.^2.0).^(1.0/2.0); # gridded wavenumber magnitudes

# cosine transform wavenumbers
kc = collect(0:Nx-1).*(pi/Lx); # for DCT-II
lc = collect(0:Ny-1).*(pi/Ly); # for DCT-II
Kc,Lc = meshgrid(kc,lc); 
Kmagc = (Kc.^2.0+Lc.^2.0).^(1.0/2.0); # gridded wavenumber magnitudes

# 1/length scale for plotting
Hks = Ks./(2.0*pi); # 1/length, equivalent to Hz for time
Hls = Ls./(2.0*pi); # 1/length, equivalent to Hz for time
Hmags = Kmags./(2.0*pi);

# wavenumber plots:

fig = figure(); CP = contourf(1:Nx,1:Ny,Kmags,200,cmap="spectral"); 
xlabel("1:N"); ylabel("1:N"); title("|K| sine"); colorbar(CP); 
savefig("./figures/sine_wavenumber_magnitude.png",format="png"); close(fig);

fig = figure(); CP = contourf(0:Nx-1,0:Ny-1,Kmagc,200,cmap="spectral"); 
xlabel("0:N-1"); ylabel("0:N-1"); title("|K| cosine"); colorbar(CP); 
savefig("./figures/cosine_wavenumber_magnitude.png",format="png"); close(fig);


# =============================================================================
# divergence by discrete sine transform of psi(x,y)

# first derivatives by dst
PSIshiftx = zeros(size(Kmags)); PSIshiftx[:,2:Nx] = PSIsinx[:,1:Nx-1]; 
PSIshifty = zeros(size(Kmags)); PSIshifty[2:Ny,:] = PSIsiny[1:Ny-1,:]; 
dpsidx = r2r(PSIshiftx.*Kc,FFTW.REDFT01,2); # DCT-III (inverse dct)
dpsidy = r2r(PSIshifty.*Lc,FFTW.REDFT01,1); # DCT-III (inverse dct)
div_psi = dpsidx+dpsidy;

# divergence error
div_error = abs(div_psiA-div_psi); max_div_error = maximum(abs(div_psiA-div_psi));
println("The maximum divergence computation error is $(max_div_error) for a $Nx by $Ny grid.\n")

# first derivative plots:

fig = figure(); CP = contourf(X./Lx,Y./Ly,abs(dpsidxA-dpsidx),200,cmap="gray");
xlabel("x"); ylabel("y"); title("x derivative, error"); 
colorbar(CP); savefig("./figures/x_derivative_error.png",format="png"); 
close(fig);

fig = figure(); CP = contourf(X./Lx,Y./Ly,abs(dpsidyA-dpsidy),200,cmap="gray");
xlabel("x"); ylabel("y"); title("y derivative, error"); 
colorbar(CP); savefig("./figures/y_derivative_error.png",format="png"); 
close(fig);

fig = figure(); CP = contourf(X./Lx,Y./Ly,abs(div_psiA-div_psi),200,cmap="gray");
xlabel("x"); ylabel("y"); title("divergence, error"); 
colorbar(CP); savefig("./figures/divergence_error.png",format="png"); 
close(fig);


# =============================================================================
# Laplacian by discrete sine transform of psi(x,y)

# the Laplacian by inverse dst
q = r2r(-PSIsin.*(Ks.^2+Ls.^2),FFTW.RODFT01,1:2); # inverse dst

# maximum error
lap_error = abs(qA-q); max_lap_error = maximum(abs(qA-q));
println("The maximum Laplacian computation error is $(max_lap_error) for a $Nx by $Ny grid.\n")

# plot of the real component, computational error 
fig = figure(); CP = contourf(X./Lx,Y./Ly,lap_error,200,cmap="gray");
xlabel("x"); ylabel("y"); title("Laplacian, error"); 
colorbar(CP); savefig("./figures/Laplacian_error.png",format="png"); 
close(fig);


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

# sine transform
Ua = r2r(ua,FFTW.RODFT10,1:2)./(4.0*Float32(Nx*Ny)); # DST-II 
Ub = r2r(ub,FFTW.RODFT10,1:2)./(4.0*Float32(Nx*Ny)); # DST-II

# aliased and de-aliased quadratic signal
u2_alias = ua.*ub; # aliased square
tic(); u2_dealias = dealias(Ua,Ub,Kmags); time = toq(); # de-aliased square
println("The computation time is for the de-aliased signal is $time seconds for a $Nx by $Ny grid.\n")

# sine transform of quadratic signal
S_alias = r2r(u2_alias,FFTW.RODFT10,1:2)./(4.0*Float32((Nx^2.0+Ny^2.0)^(1.0/2.0))); 
S1 = (abs(S_alias)); #.*2.0/float(N); 
S_dealias = r2r(u2_dealias,FFTW.RODFT10,1:2)./(4.0*Float32((Nx^2.0+Ny^2.0)^(1.0/2.0))); 
S2 = (abs(S_dealias)); #.*2.0/float(N); 

# 1D power spectrum of 2D quadratic signals
S1_vec, Hmag_vec1 = power_spectrum_2D(S1,Hmags);
S2_vec, Hmag_vec2 = power_spectrum_2D(S2,Hmags);

# plots:

fig = figure(); semilogx(Hmag_vec1,S1_vec,"r",label="aliased");
semilogx(Hmag_vec2,S2_vec,"b",label="de-aliased"); legend();
xlabel("k"); ylabel("|PSI|"); title("2D power spectrum"); 
axis([minimum(Hmag_vec1),maximum(Hmag_vec1),0.0,maximum(S1_vec)]);
savefig("./figures/quadratic_signal_power_spectrum.png",format="png"); close(fig);

fig = figure(); CP = py.contourf(X./Lx,Y./Ly,u2_alias,200,cmap="PuOr"); 
xlabel("x"); ylabel("y"); title("u*u aliased"); 
colorbar(CP); savefig("./figures/quadratic_signal_aliased.png",format="png"); 

fig = figure(); CP = py.contourf(X./Lx,Y./Ly,u2_dealias,200,cmap="PuOr"); 
xlabel("x"); ylabel("y"); title("u*u de-aliased"); 
colorbar(CP); savefig("./figures/quadratic_signal_dealiased.png",format="png");


# =============================================================================
# Poisson equation solution by dst

# Poisson equation solution: Laplacian(psi) = qA
tic(); psiP = poisson(qA,Ks,Ls); time = toq(); 
println("The first Poisson equation computation time is $(time) seconds for a $Nx by $Ny grid.\n")
Poisson_error = abs(psiP-psi); # Poisson equation solution error
max_err = maximum(Poisson_error);

fig = figure(); CP = py.contourf(X./Lx,Y./Ly,psiP,200,cmap="Spectral");
xlabel("x"); ylabel("y"); title("Laplacian(psi) = q, psi solution"); 
colorbar(CP); savefig("./figures/Poisson_solution.png",format="png"); 

fig = figure(); CP = py.contourf(X./Lx,Y./Ly,Poisson_error,200,cmap="gray")
xlabel("x"); ylabel("y"); title("Laplacian(psi) = q, psi solution error"); 
colorbar(CP); savefig("./figures/Poisson_solution_error.png",format="png"); 

println("The maximum Poisson equation computation error is $(max_err) for a $Nx by $Ny grid.\n")

# Another example: a Gaussian on a linear y slope (beta plane):
sigma = Lx/20.0; beta = 1E-9;
psi2 = exp(-((X-Lxcenter).^2.0+(Y-Lycenter).^2.0)./(2.0*sigma^2.0));
qA2 = psi2.*(((X-Lxcenter).^2.0+(Y-Lycenter).^2.0).*sigma^(-4.0)-2.0*sigma^(-2.0))-Y.*beta; 

# Poisson equation solution: Laplacian(psi) = qA
tic(); psiP2 = poisson(qA2,Ks,Ls); time = toq(); 
println("The second Poisson equation computation time is $time seconds for a $Nx by $Ny grid.\n")

fig = figure(); CP = py.contourf(X./Lx,Y./Ly,psiP2,200,cmap="Spectral");
xlabel("x"); ylabel("y"); title("Laplacian(psi) = q-By, psi solution"); 
colorbar(CP); savefig("./figures/Poisson_solution_linear_slope.png",format="png");


