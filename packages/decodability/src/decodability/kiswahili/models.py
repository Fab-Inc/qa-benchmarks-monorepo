from itertools import chain

from pydantic import BaseModel, field_validator, model_validator

from decodability.kiswahili.definitions import (
    VALID_CLUSTERS,
    VALID_GRAPHEMES,
    ClusterPattern,
    get_graphemes,
)


class KiswahiliStudentKnowledge(BaseModel):
    """Model for Kiswahili student knowledge representation."""

    graphemes: set[str] = set()
    clusters: set[str] = set()
    cluster_patterns: set[ClusterPattern] = set()
    whole_words: set[str] = set()

    @staticmethod
    def _check_lowercase(values: set[str], field_name: str) -> set[str]:
        """Raise ValueError if any string in the set is not lowercase."""
        non_lowercase = [v for v in values if v != v.lower()]
        if non_lowercase:
            raise ValueError(
                f"All {field_name} must be lowercase. Invalid values: {non_lowercase}"
            )
        return values

    @field_validator("graphemes")
    def check_known_graphemes(cls, graphemes: set[str]) -> set[str]:  # noqa: N805
        """Check that every grapheme is a valid Kiswahili grapheme."""
        cls._check_lowercase(graphemes, "graphemes")

        invalid_graphemes = [
            grapheme for grapheme in graphemes if grapheme not in VALID_GRAPHEMES
        ]

        if invalid_graphemes:
            raise ValueError(
                f"Invalid graphemes: {invalid_graphemes}. "
                "Only valid Kiswahili graphemes are allowed."
            )

        return graphemes

    @field_validator("clusters")
    def check_clusters(cls, clusters: set[str]) -> set[str]:  # noqa: N805
        """Check that all clusters are lowercase.

        Clusters are deliberately not checked against VALID_CLUSTERS: a cluster
        that follows no native pattern may still be a legitimate loan cluster
        (e.g. "br" in brashi, "st" in stesheni), so any cluster is allowed.
        """
        return cls._check_lowercase(clusters, "clusters")

    @field_validator("whole_words")
    def check_whole_words(cls, whole_words: set[str]) -> set[str]:  # noqa: N805
        """Check that all whole words are lowercase."""
        return cls._check_lowercase(whole_words, "whole_words")

    @model_validator(mode="after")
    def check_graphemes_in_clusters_known(self) -> "KiswahiliStudentKnowledge":
        """Ensure that all graphemes in known clusters are also in known graphemes."""
        graphemes_in_clusters = set(
            chain.from_iterable(get_graphemes(cluster) for cluster in self.clusters)
        )
        missing_graphemes = graphemes_in_clusters - self.graphemes

        if missing_graphemes:
            raise ValueError(
                f"Graphemes {missing_graphemes} are present in known clusters "
                "but not in known graphemes."
            )

        return self

    @model_validator(mode="after")
    def check_clusters_not_covered_by_patterns(self) -> "KiswahiliStudentKnowledge":
        """Ensure clusters are not also represented by known cluster patterns."""
        overlapping_clusters = {
            pattern: self.clusters & VALID_CLUSTERS[pattern]
            for pattern in self.cluster_patterns
            if self.clusters & VALID_CLUSTERS[pattern]
        }

        if overlapping_clusters:
            details = "; ".join(
                f"{pattern.value}: {sorted(clusters)}"
                for pattern, clusters in overlapping_clusters.items()
            )
            raise ValueError(
                "Clusters must not be specified when covered by a known cluster "
                f"pattern. Overlaps: {details}"
            )

        return self

    def __str__(self) -> str:
        """Return a string representation of the KiswahiliStudentKnowledge instance."""
        graphemes = ", ".join(f'"{g}"' for g in sorted(self.graphemes))
        clusters = ", ".join(f'"{c}"' for c in sorted(self.clusters))
        cluster_patterns = ", ".join(
            f'"{p.label}"'
            for p in sorted(self.cluster_patterns, key=lambda pattern: pattern.label)
        )
        whole_words = ", ".join(f'"{w}"' for w in sorted(self.whole_words))

        return "\n".join(
            [
                f"- Known graphemes: {graphemes or 'None'}",
                f"- Known clusters: {clusters or 'None'}",
                f"- Known cluster patterns: {cluster_patterns or 'None'}",
                f"- Known whole words: {whole_words or 'None'}",
            ]
        )
