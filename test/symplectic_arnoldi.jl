@testset "Symplectic Arnoldi" begin
    ε(x, y) = sign(y - x) / 2

    domain = 0:(n - 1)
    w(x) = 1

    skew_dot(u, v) = sum(
        ε(x, y) * w(x) * w(y) * u[x + 1] * v[y + 1]
            for x in domain, y in domain
    )

    function run_symplectic_arnoldi(orth, inplace_init)
        v₀ = InnerProductVec(fill(1 / sqrt(sum(w, domain)), n), skew_dot, norm)
        itr = ArnoldiIterator(
            u -> InnerProductVec(domain .* u[], u.dotf, u.normf),
            v₀,
            orth,
        )
        fact = initialize(itr)
        if inplace_init
            fact = initialize!(itr, fact)
        end
        for i in 1:(n - 1)
            expand!(itr, fact)
        end
        return stack(getindex, basis(fact).basis)
    end

    function max_symplectic_error(W)
        max_err = 0.0
        for i in axes(W, 2), j in axes(W, 2)
            val = skew_dot(W[:, i], W[:, j])
            if isodd(i) && j == i + 1
                max_err = max(max_err, abs(val - 1))
            elseif isodd(j) && i == j + 1
                max_err = max(max_err, abs(val + 1))
            else
                max_err = max(max_err, abs(val))
            end
        end
        return max_err
    end

    for esr in (ESR1, ESR2, ESR3)
        algs = (
            ClassicalSymplecticGramSchmidt(esr),
            ModifiedSymplecticGramSchmidt(esr),
            ClassicalSymplecticGramSchmidt2(esr),
            ModifiedSymplecticGramSchmidt2(esr),
            ClassicalSymplecticGramSchmidtIR(0.75, esr),
            ModifiedSymplecticGramSchmidtIR(0.75, esr),
        )
        for alg in algs
            @show alg
            W1 = run_symplectic_arnoldi(alg, false)
            @test max_symplectic_error(W1) < 1.0e-8

            W2 = run_symplectic_arnoldi(alg, true)
            @test W1 ≈ W2
            @test max_symplectic_error(W2) < 1.0e-8
        end
    end
end
