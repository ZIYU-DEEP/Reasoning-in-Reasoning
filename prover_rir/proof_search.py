"""Proof search using best-first search.
"""

import os
import sys
import ray
import time
import heapq
import torch
from lean_dojo import (
    Pos,
    Dojo,
    Theorem,
    LeanGitRepo,
    TacticState,
    LeanError,
    TimeoutError,
    ProofFinished,
    ProofGivenUp,
    DojoInitError,
    DojoCrashError,
    DojoHardTimeoutError,
)
from loguru import logger
from dataclasses import dataclass
from typing import List, Optional, Tuple
from ray.util.actor_pool import ActorPool

from common import zip_strict, set_logger
from common import format_input, format_output, format_suggestions
from prover.search_tree import *
from generator.model import RetrievalAugmentedGenerator, FixedTacticGenerator


@dataclass(frozen=True)
class SearchResult:
    """The result of attempting to prove a theorem."""

    theorem: Theorem
    status: Status
    proof: Optional[List[str]]

    # Some statistics during proof search.
    actor_time: float
    planner_time: float
    environment_time: float
    total_time: float
    num_total_nodes: int
    num_searched_nodes: int

    def __repr__(self):
        theorem_info = f"Theorem:\n  Repository: {self.theorem.repo}\n  File Path: {self.theorem.file_path}\n  Full Name: {self.theorem.full_name}"
        status_info = f"Status: {self.status}"

        if self.proof:
            proof_info = "Proof:\n  " + "\n  ".join(self.proof)
        else:
            proof_info = "Proof: None"

        statistics_info = f"Search Statistics:\n  Planner Time: {self.planner_time}\n  Actor Time: {self.actor_time}\n  Environment Time: {self.environment_time}\n  Total Time: {self.total_time}\n  Total Nodes: {self.num_total_nodes}\n  Searched Nodes: {self.num_searched_nodes}"

        return f"Search Result:\n\n{theorem_info}\n\n{status_info}\n\n{proof_info}\n\n{statistics_info}"


class BestFirstSearchProver:
    """A prover that uses best-first search to find proofs using a tactic generator."""

    def __init__(
        self,
        tac_gen,  # A given tactic generator.
        timeout: int,
        num_sampled_tactics: int,
        debug: bool,
        gen_type: str = "goal_driven_tactic",
        goal_gen=None,  # rir: added goal generator
        num_sampled_goals: int=10,  # rir: added num_sampled_goals
    ) -> None:

        self.tac_gen = tac_gen
        self.timeout = timeout
        self.num_sampled_tactics = num_sampled_tactics
        self.debug = debug

        self.num_expansions = 0
        self.actor_time = 0.0
        self.environment_time = 0.0
        self.total_time = None

        # RiR related
        self.gen_type = gen_type  # This will be specifically for goal_driven_tactic
        self.goal_gen = goal_gen
        self.num_sampled_goals = num_sampled_goals
        self.planner_time = 0.0

        # Set the logger (will be saved in a seprate folder)
        set_logger(verbose=self.debug, suffix=self.gen_type)
        logger.info(f"Using gen_type: {self.gen_type}.")


    def search(
        self, repo: LeanGitRepo, thm: Theorem, pos: Pos
    ) -> Optional[SearchResult]:
        logger.info(f"Proving {thm}")

        self.repo = repo
        self.theorem = thm
        self.posision = pos
        self.actor_time = 0.0
        self.environment_time = 0.0
        self.num_expansions = 0

        self.planner_time = 0.0

        if isinstance(self.tac_gen, FixedTacticGenerator):
            imps = [self.tac_gen.module]
        else:
            imps = []

        try:
            # ----------------------------------------------------------------------------
            # Init the Dojo environment
            with Dojo(thm, hard_timeout=60 + self.timeout, additional_imports=imps) as (
                dojo,
                init_state,
            ):
            # ----------------------------------------------------------------------------

                # Initialize dojo, nodes, and the priority queue
                self.dojo = dojo
                self.root = InternalNode(
                    state=init_state,
                    cumulative_logprob=0.0,  # initialize the logprob
                )
                self.nodes = {init_state: self.root}
                self.priority_queue = [self.root]

                # Do best-first search
                with torch.no_grad():
                    try:
                        self._best_first_search()
                    except DojoCrashError as ex:
                        logger.warning(f"Dojo crashed with {ex} when proving {thm}")
                        pass

            # Extract the proof
            if self.root.status == Status.PROVED:
                proof = [e.tactic for e in self.root.extract_proof()]
            else:
                proof = None

            # Record the search result
            result = SearchResult(
                theorem=thm,
                status=self.root.status,
                proof=proof,
                actor_time=self.actor_time,
                planner_time=self.planner_time,
                environment_time=self.environment_time,
                total_time=self.total_time,
                num_total_nodes=len(self.nodes),
                num_searched_nodes=self.num_expansions,
            )
            logger.info(result)
            logger.info("\n---------------------------------\n")
            return result

        except DojoInitError as ex:
            logger.warning(ex)
            return None

    def _best_first_search(self) -> None:
        time_start = time.monotonic()

        while True:
            if len(self.priority_queue) == 0:
                logger.info("Ran out of nodes to search.")
                break

            try:
                self._step()
            except DojoHardTimeoutError:
                assert time.monotonic() - time_start >= self.timeout

            self.total_time = time.monotonic() - time_start
            if self.total_time > self.timeout:
                if self.root.status == Status.PROVED:
                    logger.info("Found a proof but timed out.")
                self.root.status = Status.OPEN
                logger.info("Search timed out.")
                break

            if self.root.status == Status.FAILED:
                logger.info("Failed early!")
                break

            if self.root.status == Status.PROVED:
                logger.info("Found a proof!")
                break

    def _step(self):
        """
        Perform a single step of search.

        Selects the node with the highest priority, queries the model for suggested
        tactics, and tries each tactic in the environment, creating and enqueuing
        a new node for each valid result.
        """

        # -----------------------------------------------------------------------------
        # PREPARATION
        # Search the node with highest priority.
        search_node = heapq.heappop(self.priority_queue)
        logger.debug(f"Expanding node: {search_node}")

        # Debug with node priority
        if self.debug:
            assert all(
                search_node.priority >= node.priority for node in self.priority_queue
            )

        # Get the tactic state
        if isinstance(search_node.state, TacticState):
            ts = search_node.state.pp
        else:
            ts = search_node.state.unsolved_tactic_state
        # -----------------------------------------------------------------------------

        # -----------------------------------------------------------------------------
        # GOAL AND PROOFSTEP GENERATION
        # Generate target goals
        candidate_target_goals = self._generate_target_goals(ts)

        # Generate tactic per target goal and add them into the same queue
        # Try all tactics in order of descending logprob, and collect the results. Any
        # new nodes are added to `self.nodes`, and edges are added to the result node.
        results = []  # Initialize the result list
        for target_goal, target_goal_logprob in candidate_target_goals:

            # Low-level search on tactics
            suggestions = self._generate_tactics(ts, target_goal)  # Modified the original method

            # Check for every tactic
            for tactic, logprob in suggestions:

                # Get tactic as the edge and get the finish signal
                edge, finished = self._run_tactic(search_node,
                                                  tactic,
                                                  logprob,
                                                  target_goal_logprob,  # added for rir
                                                  )

                # Add the new edge to the results list
                results.append(edge)

                # Check if finished
                if finished: break

            # Check if finished
            if finished: break
        # -----------------------------------------------------------------------------

        # -----------------------------------------------------------------------------
        # REMAINING STUFFS
        # Store the fixed out edges of this node, marking it as explored.
        # This will trigger recursively recomputing tree statistics.
        search_node.out_edges = results
        self.num_expansions += 1

        # If we're running in debug mode, run a full test suite each step
        if self.debug:
            assert self.num_expansions == sum(
                node.is_explored
                for node in self.nodes.values()
                if isinstance(node, InternalNode)
            )
            self.check_invariants()
        # -----------------------------------------------------------------------------

    def _generate_target_goals(self, ts: str) -> List[Tuple[str, float]]:
        """
        Given current state, generate the next target state.
        """

        # ------------------------------------------------
        # MISC
        # Record the time
        t0 = time.monotonic()

        # Get the file path of the theorem
        path = str(self.theorem.file_path)

        # Set up the repo
        if self.theorem.repo != self.repo:
            path = self.theorem.repo.get_packages_dir() / self.theorem.repo.name / path

        # ------------------------------------------------
        # Format ts for RiR
        ts = format_input(ts=ts,
                          gen_type="goal")  # type is goal
        # ------------------------------------------------

        goal_suggestions = self.goal_gen.generate(
            inputs=ts,
            file_path=path,
            theorem_full_name=self.theorem.full_name,
            theorem_pos=self.posision,
            num_samples=self.num_sampled_goals,  # search width
        )

        # ------------------------------------------------
        # Format suggestions for RiR
        goal_suggestions = format_suggestions(
            suggestions=goal_suggestions,
            gen_type="goal")  # type is goal
        # ------------------------------------------------

        self.planner_time += time.monotonic() - t0

        logger.debug(f"Goal suggestions: {goal_suggestions}")
        return goal_suggestions

    def _generate_tactics(self, ts: str, target_goal: str) -> List[Tuple[str, float]]:

        # ------------------------------------------------
        # MISC
        # Record the time
        t0 = time.monotonic()

        # Get the file path of the theorem
        path = str(self.theorem.file_path)

        # Set up the repo
        if self.theorem.repo != self.repo:
            path = self.theorem.repo.get_packages_dir() / self.theorem.repo.name / path

        # ------------------------------------------------
        # Format ts for RiR
        # DEBUG
        assert self.gen_type in ["goal_driven_tactic", "default", "joint"]
        ts = format_input(ts=ts,
                          ts_=target_goal,
                          gen_type=self.gen_type)
        # ------------------------------------------------

        suggestions = self.tac_gen.generate(
            inputs=ts,
            file_path=path,
            theorem_full_name=self.theorem.full_name,
            theorem_pos=self.posision,
            num_samples=self.num_sampled_tactics,
        )

        # ------------------------------------------------
        # Format suggestions for RiR
        suggestions = format_suggestions(
            suggestions=suggestions,
            gen_type=self.gen_type)
        # ------------------------------------------------

        self.actor_time += time.monotonic() - t0

        logger.debug(f"Tactic suggestions: {suggestions}")
        return suggestions

    def _run_tactic(
        self, node: InternalNode,
        tactic: str,
        logprob: float,
        target_goal_logprob: float,  # added for rir
    ) -> Tuple[Edge, bool]:

        # Get the current time
        t0 = time.monotonic()

        # Use LeanDojo to get response
        response = self.dojo.run_tac(node.state, tactic)

        # Record the time
        elapsed = time.monotonic() - t0
        self.environment_time += elapsed

        # ---------------------------------------------------------------
        # Set the result node
        try:
            # If we've seen this response before, use the existing node
            result_node = self.nodes[response]

        except KeyError:
            # Build a new node
            if isinstance(response, ProofFinished):
                result_node = ProofFinishedNode(response)

            elif type(response) in (
                LeanError,
                TimeoutError,
                ProofGivenUp,
            ):
                result_node = ErrorNode(response)

            else:
                assert isinstance(response, TacticState)
                result_node = InternalNode(
                    state=response,
                    # Add the prob for p(y | s, s*) and p(s*| s)
                    cumulative_logprob=node.cumulative_logprob + logprob + target_goal_logprob,
                )

            if result_node.status == Status.OPEN:  # Don't search proved/failed nodes
                heapq.heappush(self.priority_queue, result_node)  # type: ignore

        # Record the new node and add it to the search queue.
        self.nodes[response] = result_node
        # ---------------------------------------------------------------

        # ---------------------------------------------------------------
        # Build an edge connecting these nodes.
        # Will be added to the source node externally.
        edge = Edge(tactic=tactic,
                    src=node,
                    dst=result_node)
        # ---------------------------------------------------------------

        if isinstance(result_node, InternalNode):
            result_node.in_edges.append(edge)

        return edge, isinstance(response, ProofFinished)

    #########
    # DEBUG #
    #########

    def check_invariants(self):
        """Perform some sanity checks."""
        for node in self.priority_queue:
            assert node in self.nodes.values()
            assert isinstance(node, InternalNode)
            assert not node.is_explored

        for response, node in self.nodes.items():
            if isinstance(response, ProofFinished):
                assert isinstance(node, ProofFinishedNode)
                assert node not in self.priority_queue
                assert self.root.status == Status.PROVED
            elif type(response) in (
                LeanError,
                TimeoutError,
                ProofGivenUp,
            ):
                assert isinstance(node, ErrorNode)
                assert node not in self.priority_queue
            else:
                assert isinstance(node, InternalNode)

                if node.is_explored:
                    assert node not in self.priority_queue
                else:
                    assert node in self.priority_queue

                node.check_invariants()


@ray.remote
class CpuProver(BestFirstSearchProver):
    """Ray actor for running an instance of `BestFirstSearchProver` on a CPU."""

    def __init__(
        self,
        ckpt_path: Optional[str],
        indexed_corpus_path: Optional[str],
        tactic: Optional[str],
        module: Optional[str],
        timeout: int,
        num_sampled_tactics: int,
        debug: bool,
        gen_type: str="goal_driven_tactic",
        goal_ckpt_path: Optional[str] = None,  # added for rir
        num_sampled_goals: int = 10,  # added for rir
    ) -> None:
        # ----------------------------------------------------------------------
        # SETUP THE TAC_GEN
        if ckpt_path is None:
            tac_gen = FixedTacticGenerator(tactic, module)
        else:
            tac_gen = RetrievalAugmentedGenerator.load(
                ckpt_path=ckpt_path,
                device=torch.device("cpu"),
                freeze=True,
            )
            if tac_gen.retriever is not None:
                if indexed_corpus_path is not None:
                    tac_gen.retriever.load_corpus(indexed_corpus_path)
                tac_gen.retriever.reindex_corpus(batch_size=32)
        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------
        # SETUP THE GOAL_GEN
        if goal_ckpt_path is None:
            goal_gen = FixedTacticGenerator(tactic, module)
        else:
            goal_gen = RetrievalAugmentedGenerator.load(
                ckpt_path=goal_ckpt_path,
                device=torch.device("cpu"),
                freeze=True,
            )

            if goal_gen.retriever is not None:
                if indexed_corpus_path is not None:
                    goal_gen.retriever.load_corpus(indexed_corpus_path)
                goal_gen.retriever.reindex_corpus(batch_size=32)
        # ----------------------------------------------------------------------

        super().__init__(
            tac_gen,
            timeout,
            num_sampled_tactics,
            debug,
            gen_type,
            goal_gen,  # added for rir
            num_sampled_goals,  # added for rir
        )


@ray.remote(num_gpus=1)
class GpuProver(BestFirstSearchProver):
    """Ray actor for running an instance of `BestFirstSearchProver` on a GPU."""

    def __init__(
        self,
        ckpt_path: Optional[str],
        indexed_corpus_path: Optional[str],
        tactic: Optional[str],
        module: Optional[str],
        timeout: int,
        num_sampled_tactics: int,
        debug: bool,
        gen_type: str="goal_driven_tactic",
        goal_ckpt_path: Optional[str] = None,  # added for rir
        num_sampled_goals: int = 10,  # added for rir
    ) -> None:

        # ----------------------------------------------------------------------
        # SETUP THE TAC_GEN
        if ckpt_path is None:
            tac_gen = FixedTacticGenerator(tactic, module)
        else:
            tac_gen = RetrievalAugmentedGenerator.load(
                ckpt_path=ckpt_path,
                device=torch.device("cuda"),
                freeze=True,
            )
            if tac_gen.retriever is not None:
                if indexed_corpus_path is not None:
                    tac_gen.retriever.load_corpus(indexed_corpus_path)
                tac_gen.retriever.reindex_corpus(batch_size=32)
        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------
        # SETUP THE GOAL_GEN
        if goal_ckpt_path is None:
            goal_gen = FixedTacticGenerator(tactic, module) 
        else:
            goal_gen = RetrievalAugmentedGenerator.load(
                ckpt_path=goal_ckpt_path,
                device=torch.device("cuda"),
                freeze=True,
            )

            if goal_gen.retriever is not None:
                if indexed_corpus_path is not None:
                    goal_gen.retriever.load_corpus(indexed_corpus_path)
                goal_gen.retriever.reindex_corpus(batch_size=32)
        # ----------------------------------------------------------------------

        super().__init__(
            tac_gen,
            timeout,
            num_sampled_tactics,
            debug,
            gen_type,
            goal_gen,  # added for rir
            num_sampled_goals,  # added for rir
        )


class DistributedProver:
    """A distributed prover that uses Ray to parallelize the proof search.

    It is a wrapper around `CpuProver` and `GpuProver` that handles the different
    devices and different number of concurrent provers.
    """

    def __init__(
        self,
        ckpt_path: Optional[str],
        indexed_corpus_path: Optional[str],
        tactic: Optional[str],
        module: Optional[str],
        num_workers: int,
        num_gpus: int,
        timeout: int,
        num_sampled_tactics: int,
        debug: Optional[bool] = False,
        gen_type: str="goal_driven_tactic",
        goal_ckpt_path: Optional[str] = None,  # added for rir
        num_sampled_goals: int = 10,  # added for rir
    ) -> None:

        # --------------------------------------------------
        if ckpt_path is None:
            assert tactic and not indexed_corpus_path
        else:
            assert not tactic and not module
        # --------------------------------------------------

        self.distributed = num_workers > 1

        if not self.distributed:

            # ------------------------------------------------------------------------
            # TAC_GEN
            if ckpt_path is None:
                tac_gen = FixedTacticGenerator(
                    tactic=tactic,
                    module=module)
            else:
                device = torch.device("cuda") if num_gpus > 0 else torch.device("cpu")
                tac_gen = RetrievalAugmentedGenerator.load(
                    ckpt_path=ckpt_path,
                    device=device,
                    freeze=True,
                    # gen_type=gen_type,  # adding gen_type for RiR
                )
                if tac_gen.retriever is not None:
                    assert indexed_corpus_path is not None
                    tac_gen.retriever.load_corpus(indexed_corpus_path)
             # ------------------------------------------------------------------------

            # ------------------------------------------------------------------------
            # TAC_GEN
            if goal_ckpt_path is None:
                goal_gen = FixedTacticGenerator(
                    tactic=tactic,
                    module=module)
            else:
                device = torch.device("cuda") if num_gpus > 0 else torch.device("cpu")
                goal_gen = RetrievalAugmentedGenerator.load(
                    ckpt_path=goal_ckpt_path,
                    device=device,
                    freeze=True,
                    # gen_type=gen_type,  # adding gen_type for RiR
                )
                if goal_gen.retriever is not None:
                    assert indexed_corpus_path is not None
                    goal_gen.retriever.load_corpus(indexed_corpus_path)
             # ------------------------------------------------------------------------

            self.prover = BestFirstSearchProver(
                tac_gen=tac_gen,
                timeout=timeout,
                num_sampled_tactics=num_sampled_tactics,
                debug=debug,
                gen_type=gen_type,  # Add gen_type for RiR
                goal_gen=goal_gen,  # added for rir
                num_sampled_goals=num_sampled_goals,
            )
            return

        if num_gpus >= 1:
            logger.info(f"Launching {num_workers} workers with {num_gpus} GPUs.")
            num_gpus_per_worker = num_gpus / num_workers
            provers = [
                GpuProver.options(num_gpus=num_gpus_per_worker).remote(
                    ckpt_path,
                    indexed_corpus_path,
                    tactic,
                    module,
                    timeout=timeout,
                    num_sampled_tactics=num_sampled_tactics,
                    debug=debug,
                    gen_type=gen_type,  # Add gen_type for RiR
                    goal_ckpt_path=goal_ckpt_path,  # added for rir
                    num_sampled_goals=num_sampled_goals,
                )
                for _ in range(num_workers)
            ]
        else:
            logger.info(f"Launching {num_workers} CPU workers.")
            provers = [
                CpuProver.remote(
                    ckpt_path,
                    indexed_corpus_path,
                    tactic,
                    module,
                    timeout=timeout,
                    num_sampled_tactics=num_sampled_tactics,
                    debug=debug,
                    gen_type=gen_type,
                    goal_ckpt_path=goal_ckpt_path,  # added for rir
                    num_sampled_goals=num_sampled_goals,
                )
                for _ in range(num_workers)
            ]

        self.prover_pool = ActorPool(provers)

    def search_unordered(
        self, repo: LeanGitRepo, theorems: List[Theorem], positions: List[Pos]
    ) -> List[Optional[SearchResult]]:
        """Parallel proof search for `theorems`. The order of the results is not guaranteed to match the order of the input."""
        if not self.distributed:
            return [
                self.prover.search(repo, thm, pos)
                for thm, pos in zip_strict(theorems, positions)
            ]

        try:
            results = list(
                self.prover_pool.map_unordered(
                    lambda p, x: p.search.remote(repo, x[0], x[1]),
                    zip_strict(theorems, positions),
                )
            )
        except ray.exceptions.RayActorError as ex:
            logger.error(ex)
            sys.exit(1)

        return results
