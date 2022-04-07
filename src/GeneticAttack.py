import torch
from src.trainable import TorchTrainable


# Attacker algorithm here
class GeneticAttack:

    def __init__(self, x: torch.Tensor, y: int, trainable: TorchTrainable, N=10, epochs=50, selective_pressure=0.2,
                 asexual_repro=1, epsilon=0.1, uncertainty_power=2, sameness_power=4):

        """
        :param x: 28 by 28 torch tensor of the original image.
        :param y: Target.
        :param trainable: Trainable targeted by the attack.
        :param N; Size of the population during the simulation.
        :param epochs: Number of epochs to run the genetic algorithm.
        :param elite_perc: Percentage of the most fit population that are considered during reproduction.
        :param epsilon: Coefficient for the linear combination between uncertainty and sameness in the loss.
        :param uncertainty_power: Power of the exponent used in the uncertainty equation.
        :param sameness_power: Power of the exponent used in the sameness equation.
        :return:
        """
        self.x = x
        self.y = y
        self.trainable = trainable
        self.N = N
        self.epochs = epochs
        self.selective_pressure = selective_pressure
        self.asexual_repro = asexual_repro
        self.epsilon = epsilon
        self.uncertainty_power = uncertainty_power
        self.sameness_power = sameness_power

        # Create a population by duplicating the attack target (x)
        population = torch.stack([x for i in range(N)])

        for i in range(epochs):

            # TODO: add different types of mutations
            # TODO: implement perturbation decay

            # Evaluate the quality of the population
            qual = self.evaluate_quality(population)
            rank = torch.argsort(qual, descending=True)  # Inverse the rank (xform max prob into min problem)

            if i % 10:
                print(qual[rank[-1]].data)

            # Choose the fittest units for reproduction (N/2 parents chosen with replacement among the fittest)
            parents_idx = []

            for n in range(self.N // 2):
                parents = self.select_parents(rank)
                parents_idx.append(parents)

            parents_idx = torch.stack(parents_idx)

            # Create the new generation from the fittest parents
            children = self.generate_children(population, parents_idx)

            # Perturb the population with random mutations (non zero values only)
            children[torch.where(children != 0)] += torch.normal(0, 0.05, size=children[torch.where(children != 0)].shape)
            children = torch.clamp(children, 0, 1)

            # Elitism (maintain top solution at all times) # TODO: DEBUG ELITISM
            top_solution = population[rank[-1]]
            population = children
            population[0] = top_solution

            a=10

    def complement_idx(self, idx, dim):

        # SOURCE: https://stackoverflow.com/questions/67157893/pytorch-indexing-select-complement-of-indices

        """
        Compute the complement: set(range(dim)) - set(idx).
        idx is a multi-dimensional tensor, find the complement for its trailing dimension,
        all other dimension is considered batched.
        Args:
            idx: input index, shape: [N, *, K]
            dim: the max index for complement
        """
        a = torch.arange(dim, device=idx.device)
        ndim = idx.ndim
        dims = idx.shape
        n_idx = dims[-1]
        dims = dims[:-1] + (-1,)
        for i in range(1, ndim):
            a = a.unsqueeze(0)
        a = a.expand(*dims)
        masked = torch.scatter(a, -1, idx, 0)
        compl, _ = torch.sort(masked, dim=-1, descending=False)
        compl = compl.permute(-1, *tuple(range(ndim - 1)))
        compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))
        return compl

    def generate_children(self, population, parents_idx):

        parent_pool = torch.clone(parents_idx)

        # Asexual reproduction (cloning)
        repro_mask = torch.bernoulli(torch.Tensor([self.asexual_repro for n in range(self.N//2)]))
        mask_idx = torch.where(repro_mask == 1)[0]  # Apply mask and find parents to clone
        clones = population[parents_idx[mask_idx]]
        clones = torch.flatten(clones, start_dim=0, end_dim=1)

        parent_pool = parent_pool[self.complement_idx(mask_idx, dim=self.N//2)]

        # Reshape the parents index tensor in preparation for ''fancy'' indexing
        inv_parents_idx = parent_pool.resize(parent_pool.shape[1], parent_pool.shape[0])

        # Sexual reproduction (gene sharing)
        r = torch.rand(size=self.x.shape)
        batch1 = r * population[inv_parents_idx[0]] + (1 - r) * population[inv_parents_idx[1]]
        batch2 = r * population[inv_parents_idx[1]] + (1 - r) * population[inv_parents_idx[0]]
        children = torch.cat([batch1, batch2])

        children = torch.cat([clones, children])

        return children

    def select_parents(self, rank):

        #TODO: optimize (naive implementation)

        lower_bound = int((1 - self.selective_pressure) * self.N)
        first_index = torch.randint(low=lower_bound, high=self.N, size=(1,), device=self.trainable.device)

        # Choose best parent randomly according to selective pressure
        first_parent = rank[first_index]

        # Choose second parent
        second_parent = first_parent
        while second_parent == first_parent:
            second_parent = torch.randint(0, self.N - 1, size=(1,), device=self.trainable.device)

        return torch.tensor([first_parent, second_parent], device=self.trainable.device)

    def evaluate_quality(self, adversarial_x):

        """
        :param adversarial_x: batch of 28 by 28 perturbed images
        :return:
        """
        # TODO: Find optimal parameters for quality eval (epsilon, powers)

        uncertainty_loss = self.trainable(adversarial_x)[:, self.y]

        sameness_loss = (self.x-adversarial_x)**self.sameness_power

        if len(adversarial_x.shape) == 2:
            sameness_loss = torch.sum((self.x-adversarial_x)**self.sameness_power).to(self.trainable.device)

        else:
            for x in range(len(adversarial_x.shape)-1):
                sameness_loss = torch.sum(sameness_loss, dim=1)

        sameness_loss = sameness_loss.to(self.trainable.device)

        return uncertainty_loss + self.epsilon * sameness_loss
