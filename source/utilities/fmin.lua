require "math"

function fmin(func, a, b, tol)
	local phi = (-1 + math.sqrt(5)) / 2

	local x1  = phi * (b - a) + a
	local x2  = b - phi * (b - a)
	local fx1 = func(x1)
	local fx2 = func(x2)

	while b - a >= tol do
		if fx1 <= fx2 then
			b = x2
			x2 = x1
			fx2 = fx1

			x1 = phi * (b - a) + a
			fx1 = func(x1)
		else
			a = x1
			x1 = x2
			fx1 = fx2

			x2 = b - phi * (b - a)
			fx2 = func(x2)
		end
	end

	return a, b
end
