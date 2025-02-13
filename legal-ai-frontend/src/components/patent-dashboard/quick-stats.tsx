import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { FileText, AlertTriangle, TrendingUp } from "lucide-react"

interface QuickStatsProps {
  data: {
    temporal_analysis: {
      total_applications: number
    }
    rejection_analysis: {
      total_rejections: number
    }
  }
}

export function QuickStats({ data }: QuickStatsProps) {
  const successRate = Math.round(
    (1 - data.rejection_analysis.total_rejections / data.temporal_analysis.total_applications) * 100
  )

  return (
    <div className="grid gap-4 md:grid-cols-3">
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Total Applications</CardTitle>
          <FileText className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{data.temporal_analysis.total_applications.toLocaleString()}</div>
          <p className="text-xs text-muted-foreground">
            Total patent applications analyzed
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Total Rejections</CardTitle>
          <AlertTriangle className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{data.rejection_analysis.total_rejections.toLocaleString()}</div>
          <p className="text-xs text-muted-foreground">
            Total rejection instances
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Success Rate</CardTitle>
          <TrendingUp className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{successRate}%</div>
          <p className="text-xs text-muted-foreground">
            Applications without rejections
          </p>
        </CardContent>
      </Card>
    </div>
  )
}